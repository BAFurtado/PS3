# calibration_conf.py

# Parameters to calibrate: [lower_bound, upper_bound]
# Behavioral parameters calibrated on anchor region only.
# All other capitals inherit these values; only structural
# data inputs (A' matrix, ε_s, distributions) vary by region.
#
# Cut from 11 to 5 on 2026-09-28 (see private/notes/text_emissions/MSG_GUSTAVO_2026-09-28.md).
# Held at their conf/default/params.py values:
#   MARKUP, RELEVANCE_UNEMPLOYMENT_SALARIES  low S_Ti in the rescored 08-17 archive
#   PRICE_RUGGEDNESS, INVENTORY_TARGET_RATIO no fitness moment constrains them
#   NATURAL_SEPARATION_RATE                  set from published turnover figures, not fitted
#   ENVIRONMENTAL_EFFICIENCY_STEP            no macro moment constrains it; separate 1-D stage on emissions
# Ranges are narrowed around the defaults, using the 08-17 archive only as loose guidance (a third of its box
# was explosive, and it predates the labour-matching fixes). Wave 1 exists to narrow them further.
CALIBRATION_PARAMETERS = {

    # Production
    "PRODUCTIVITY_MAGNITUDE_DIVISOR": [0.5,   1.5],   # default: 1.0
    "PRODUCTIVITY_EXPONENT":          [0.5,   0.8],   # default: 0.65

    # Pricing
    "STICKY_PRICES":                  [0.3,   0.9],   # default: 0.7

    # Labor market
    "LABOR_MARKET":                   [0.4,   0.9],   # default: 0.8
    "PCT_DISTANCE_HIRING":            [0.05,  0.4],   # default: 0.2; commuting term live only since ceeb0aa

}

CALIBRATION_SETTINGS = {

    # "lhs": Latin hypercube of `samples` sets, for history matching (the default).
    # "sobol": Saltelli design, N * (k + 2) sets; use powers of 2 and N >= 512 for stable S_Ti.
    "design":         "lhs",
    "samples":        64,
    "runs_per_sample": 2,   # seeds per set; >= 2 needed for noise weights and implausibility
    "lhs_seed":       42,

    # Burn-in excluded from fitness; moments computed over [burn_in_end, target_end_year]
    "burn_in_end":       "2012-01-01",
    "target_start_year": "2010-01-01",
    "target_end_year":   "2025-01-01",

    # Anchor region for calibration
    "calibration_region": "BELO HORIZONTE",

    # Moments scored. Left out: inflation_mean (the model has no nominal anchor, no set can reach it) and the
    # Gini (the model measures household permanent income, the observed series is per-capita household
    # income for the whole state; gini_mean can be added back once a comparable target exists).
    "fitness_moments": ["gdp_growth_mean", "gdp_growth_std", "unemployment_mean", "unemployment_std",
                        "inflation_std"],
    # "inverse_noise": each relative deviation weighted by |obs| / sqrt(seed_sd^2 + (model_discrepancy * obs)^2),
    # i.e. near equal, down-weighting moments that are noisy across seeds; "equal": 1 / number of moments.
    "fitness_weights": "inverse_noise",

    # History matching: a set is implausible when, for any moment,
    # |sim - obs| / sqrt(seed_sd^2 + (model_discrepancy * obs)^2) > implausibility_cutoff.
    "model_discrepancy":     0.10,
    "implausibility_cutoff": 3.0,

    # Sobol / SALib settings
    "sobol_calc_second_order": False,
    "sobol_seed":              42,

    "observed_data_path": 'analysis/calibration/data/observed_bh.csv',

    # Parameters with S_Ti (or |rho| in fallback mode) below this threshold are candidates to freeze
    "freeze_threshold_sti": 0.05,
}
