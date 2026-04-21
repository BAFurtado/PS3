"""
Emissions policy runner: 5 scenarios × all Brazilian capitals.

The burn-in period (before ECO_POLICY_DAYS) serves as the implicit baseline
for every run — no separate no-policy scenario is needed.

Active scenarios
----------------
  TAX              — flat emission tax only          (TAX_EMISSION=10)
  SUBSIDIES        — eco-investment subsidies only   (ECO_INVESTMENT_SUBSIDIES=0.2)
  BOTH             — tax + subsidies combined

Stubbed scenarios (implement before enabling)
---------------------------------------------
  TARGETED_SUBSIDIES — sector-weighted subsidies proportional to emission
                       intensity. Touch point: firm.decision_on_eco_efficiency()
                       in agents/firm.py — replace the flat ECO_INVESTMENT_SUBSIDIES
                       rate with a per-sector lookup derived from emissions_base.
                       New param required: TARGETED_SUBSIDIES (bool).

  TAX_RECYCLING      — emission tax revenue redistributed to regional households
                       as a lump-sum transfer each month. Touch points:
                       (1) simulation.py monthly loop — add redistribution step
                           after create_externalities(), draining
                           region.cumulative_treasure['emissions'] each month.
                       (2) region.py or funds.py — add redistribution helper.
                       New param required: TAX_RECYCLING (bool).

Usage
-----
    python runner_emissions.py
"""
import logging
import pathlib
import datetime

from main import multiple_runs, gen_output_dir

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO)

# ── Configuration ─────────────────────────────────────────────────────────────

RUNS = 5
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

SCENARIOS = {
    'TAX':      {'TAX_EMISSION': 10, 'ECO_INVESTMENT_SUBSIDIES': 0},
    'SUBSIDIES': {'TAX_EMISSION': 0,  'ECO_INVESTMENT_SUBSIDIES': 0.2},
    'BOTH':     {'TAX_EMISSION': 10, 'ECO_INVESTMENT_SUBSIDIES': 0.2},

    # Uncomment once implemented — see module docstring for touch points
    # 'TARGETED_SUBSIDIES': {'TARGETED_SUBSIDIES': True, 'TAX_EMISSION': 0,
    #                        'ECO_INVESTMENT_SUBSIDIES': 0.2},
    # 'TAX_RECYCLING':      {'TAX_RECYCLING': True, 'TAX_EMISSION': 10,
    #                        'ECO_INVESTMENT_SUBSIDIES': 0},
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
