"""
Emissions policy runner: 8 scenarios × all Brazilian capitals.

The burn-in period (before ECO_POLICY_DAYS) serves as the implicit baseline
for every run — no separate no-policy scenario is needed.

Scenarios
---------
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
import logging
import pathlib

from main import multiple_runs, gen_output_dir

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
    'SALVADOR',
    'SAO LUIS',
    'TERESINA',
    'VITORIA',
]

# Each scenario explicitly sets all policy flags to avoid inheriting defaults.
_BASE = {'TARGETED_SECTORS': ['Agriculture', 'Transport', 'Utilities'],
         'CARBON_RECYCLING_QUANTILE': 0.25}

SCENARIOS = {
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

def main():
    overrides = [
        {**scenario_params, 'PROCESSING_ACPS': [capital]}
        for capital in CAPITAIS
        for scenario_params in SCENARIOS.values()
    ]

    n_total = len(overrides) * RUNS
    logger.info(
        f"Emissions runner: {len(SCENARIOS)} scenario(s) × {len(CAPITAIS)} capital(s) "
        f"× {RUNS} run(s) = {n_total} simulations"
    )

    output_dir = gen_output_dir('emissions')
    multiple_runs(overrides, RUNS, CPUS, output_dir)
    logger.info("All emissions runs completed.")


if __name__ == '__main__':
    main()
