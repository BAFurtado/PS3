"""
Emissions policy runner: 9 scenarios × all Brazilian capitals.

NO_POLICY runs with every policy flag off for the entire simulation (not just
pre-ECO_POLICY_DAYS) and serves as the never-treated control arm for the
DiD / event-study comparison. The other 8 scenarios still have their own
implicit pre-ECO_POLICY_DAYS baseline for within-run before/after comparison.

Scenarios
---------
  NO_POLICY                    — control: no policy for the full run (DiD/event-study counterfactual)
  TAX                          — flat emission tax only
  SUBSIDIES                    — flat eco-investment subsidies only
  TARGETED_SUBSIDIES           — sector-weighted subsidies (Agriculture, Transport, Utilities)
  TAX + SUBSIDIES              — tax + flat subsidies
  TAX + TARGETED_SUBSIDIES     — tax + sector-weighted subsidies
  TAX_RECYCLING                — tax with revenue recycled to bottom-quartile households
  TAX_RECYCLING + SUBSIDIES    — recycling + flat subsidies
  TAX_RECYCLING + TARGETED     — recycling + sector-weighted subsidies

Usage
-----
    python runner_emissions.py
"""
import datetime
import logging
import os
import pathlib
from collections import Counter

import conf
import main_plotting
from checkpoint import pending_jobs
from main import multiple_runs, gen_output_dir, _run_jobs_parallel

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO)

# ── Configuration ─────────────────────────────────────────────────────────────

RUNS = 15
CPUS = 8

LOG_DIR = pathlib.Path("logs/emissions")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CAPITAIS = [
    'ARACAJU',
    'BELEM',
    'BELO HORIZONTE',
    'BOA VISTA',
    'BRASILIA',
    'CAMPO GRANDE',
    'CUIABA',
    'CURITIBA',
    'FLORIANOPOLIS',
    'FORTALEZA',
    'GOIANIA',
    'JOAO PESSOA',
    'MACAPA',
    'MACEIO',
    'MANAUS',
    'NATAL',
    'PALMAS',
    'PORTO ALEGRE',
    'PORTO VELHO',
    'RECIFE',
    'RIO BRANCO',
    'RIO DE JANEIRO',
    'SALVADOR',
    'SAO LUIS',
    'SAO PAULO',
    'TERESINA',
    'VITORIA',
]

# TARGETED_SECTORS/CARBON_RECYCLING_QUANTILE are left out here (never vary, and
# adding them pushed job output paths past Windows' MAX_PATH) — relied on from
# conf/default/params.py instead; asserted in main() below.
_BASE = {}

SCENARIOS = {
    # Control arm: all policy flags off for the full run (burn-in AND
    # post-trigger), not just pre-ECO_POLICY_DAYS. Serves as the
    # never-treated counterfactual for the DiD / event-study comparison.
    'NO_POLICY': {
        **_BASE,
        'TAX_EMISSION': 0, 'ECO_INVESTMENT_SUBSIDIES': 0,
        'TARGETED_SUBSIDIES': False, 'CARBON_TAX_RECYCLING': False,
    },
    'TAX': {
        **_BASE,
        'TAX_EMISSION': 10, 'ECO_INVESTMENT_SUBSIDIES': 0,
        'TARGETED_SUBSIDIES': False, 'CARBON_TAX_RECYCLING': False,
    },
    'SUBSIDIES': {
        **_BASE,
        'TAX_EMISSION': 0, 'ECO_INVESTMENT_SUBSIDIES': 0.2,
        'TARGETED_SUBSIDIES': False, 'CARBON_TAX_RECYCLING': False,
    },
    'TARGETED_SUBSIDIES': {
        **_BASE,
        'TAX_EMISSION': 0, 'ECO_INVESTMENT_SUBSIDIES': 0.2,
        'TARGETED_SUBSIDIES': True,  'CARBON_TAX_RECYCLING': False,
    },
    'TAX_SUBSIDIES': {
        **_BASE,
        'TAX_EMISSION': 10, 'ECO_INVESTMENT_SUBSIDIES': 0.2,
        'TARGETED_SUBSIDIES': False, 'CARBON_TAX_RECYCLING': False,
    },
    'TAX_TARGETED_SUBSIDIES': {
        **_BASE,
        'TAX_EMISSION': 10, 'ECO_INVESTMENT_SUBSIDIES': 0.2,
        'TARGETED_SUBSIDIES': True,  'CARBON_TAX_RECYCLING': False,
    },
    'TAX_RECYCLING': {
        **_BASE,
        'TAX_EMISSION': 10, 'ECO_INVESTMENT_SUBSIDIES': 0,
        'TARGETED_SUBSIDIES': False, 'CARBON_TAX_RECYCLING': True,
    },
    'TAX_RECYCLING_SUBSIDIES': {
        **_BASE,
        'TAX_EMISSION': 10, 'ECO_INVESTMENT_SUBSIDIES': 0.2,
        'TARGETED_SUBSIDIES': False, 'CARBON_TAX_RECYCLING': True,
    },
    'TAX_RECYCLING_TARGETED': {
        **_BASE,
        'TAX_EMISSION': 10, 'ECO_INVESTMENT_SUBSIDIES': 0.2,
        'TARGETED_SUBSIDIES': True,  'CARBON_TAX_RECYCLING': True,
    },
}

# ── Runner ────────────────────────────────────────────────────────────────────

def _configure_file_logging() -> pathlib.Path:
    """Persist orchestration-level logs (retries, failures, summary) to a file.

    Per-simulation internal logging still only reaches the console when run
    with CPUS=1 — worker processes don't inherit this handler.
    """
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"run_{ts}.log"
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logger.info(f"Logging to {log_file}")
    return log_file


def main():
    assert conf.PARAMS['TARGETED_SECTORS'] == ['Agriculture', 'Transport', 'Utilities'], \
        "conf/default/params.py TARGETED_SECTORS drifted from what this runner assumes"
    assert conf.PARAMS['CARBON_RECYCLING_QUANTILE'] == 0.25, \
        "conf/default/params.py CARBON_RECYCLING_QUANTILE drifted from what this runner assumes"

    # Needed for firm-level panel; scoped here since conf/run.py is gitignored.
    conf.RUN['SAVE_DATA'] = ['firms']

    log_file = _configure_file_logging()

    # job_group maps each override's path-fragment (main_plotting.conf_to_str)
    # back to (capital, scenario_name), so incomplete jobs can be reported
    # in human-readable terms rather than raw parameter dumps.
    overrides = []
    job_group = {}
    for capital in CAPITAIS:
        for scenario_name, scenario_params in SCENARIOS.items():
            o = {**scenario_params, 'PROCESSING_ACPS': [capital]}
            overrides.append(o)
            job_group[main_plotting.conf_to_str(o)] = (capital, scenario_name)

    n_total = len(overrides) * RUNS
    logger.info(
        f"Emissions runner: {len(SCENARIOS)} scenario(s) × {len(CAPITAIS)} capital(s) "
        f"× {RUNS} run(s) = {n_total} simulations"
    )

    output_dir = gen_output_dir('emissions')
    multiple_runs(overrides, RUNS, CPUS, output_dir)

    pending, cleaned = pending_jobs(output_dir)
    if pending:
        logger.warning(
            f"{len(pending)}/{n_total} job(s) incomplete after the initial pass "
            f"(cleaned {cleaned} partial dir(s)). Retrying once..."
        )
        _run_jobs_parallel(pending, CPUS)
        pending, _ = pending_jobs(output_dir)

    _log_summary(n_total, pending, job_group, output_dir)
    logger.info(f"Full log: {log_file}")


def _log_summary(n_total, pending, job_group, output_dir):
    n_ok = n_total - len(pending)
    sep = "=" * 60
    logger.info(sep)
    logger.info("EMISSIONS RUNNER SUMMARY")
    logger.info(sep)
    logger.info(f"{n_ok}/{n_total} simulation(s) completed successfully.")

    if pending:
        counts = Counter()
        for job in pending:
            key = os.path.basename(os.path.dirname(job['path']))
            counts[job_group.get(key, ('UNKNOWN', key))] += 1
        logger.error(f"{len(pending)} simulation(s) still incomplete after retry:")
        for (capital, scenario_name), n in sorted(counts.items()):
            logger.error(f"  [FAIL] {capital:<20s} {scenario_name:<28s} {n} run(s) missing")
        logger.error(f'Resume manually with: python main.py resume "{output_dir}" -c {CPUS}')
    else:
        logger.info("All emissions runs completed successfully.")
    logger.info(sep)


if __name__ == '__main__':
    main()
