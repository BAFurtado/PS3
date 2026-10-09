# calibration_conf.py

# Parameters to calibrate: [lower_bound, upper_bound]. One set of values for every region; regions differ only by
# their data. Every other parameter stays at its conf/default/params.py value, among them MARKUP, PRICE_RUGGEDNESS,
# INVENTORY_TARGET_RATIO, HOUSING_FINANCIAL_WEIGHT and CONSTRUCTION_PLAN; ENVIRONMENTAL_EFFICIENCY_STEP is a separate
# 1-D stage on emissions. PRODUCTIVITY_MAGNITUDE_DIVISOR is set from IBGE municipal value added at start-up and the
# labour flows come from PME 2010 (LABOUR_FLOWS 'data'), so neither is calibrated.
CALIBRATION_PARAMETERS = {

    # Production
    "PRODUCTIVITY_EXPONENT":       [0.5,   0.8],   # default: 0.65

    # Pricing
    "STICKY_PRICES":               [0.3,   0.9],   # default: 0.7

    # Labor market
    "LABOR_MARKET":                [0.4,   0.9],   # default: 0.8

    # Housing
    "BUILD_VACANCY_SENSITIVITY":   [7,     19],    # default: 13

}

# Parameters with a few options: each set takes one, from a dimension of its own in the Latin hypercube, so the options
# are drawn in equal numbers. None in this wave.
CALIBRATION_OPTIONS = {}

CALIBRATION_SETTINGS = {

    # "lhs": Latin hypercube of `samples` sets, for history matching (the default).
    # "sobol": Saltelli design, N * (k + 2) sets; use powers of 2 and N >= 512 for stable S_Ti (no options).
    "design":          "lhs",
    "samples":         32,
    "runs_per_sample": 2,    # seeds per set and region; >= 2 needed for the seed noise in the implausibility
    "lhs_seed":        42,
    "seed_base":       1000, # model seed of replication i is seed_base + i, the same in every set and region

    # Simulated period
    "target_start_year": "2010-01-01",
    "target_end_year":   "2020-01-01",

    # Every set runs in every region
    "calibration_regions": ["BELO HORIZONTE", "FORTALEZA", "GOIANIA", "PALMAS"],

    # "levels": the levels of data/level_targets.csv, means of the last levels_window months (calibration/levels.py);
    # a set is implausible when any run fails a hard constraint (levels.explodes) or, for any region and level,
    # its seed mean lies outside the band by more than implausibility_cutoff x sqrt(seed_sd^2 +
    # (model_discrepancy x band midpoint)^2).
    # "series": Belo Horizonte's moments of GDP growth, unemployment and inflation over [burn_in_end, target_end_year]
    # (one region, BELO HORIZONTE, and fitness_moments below).
    "targets":        "levels",
    "levels_window":  36,

    "model_discrepancy":     0.10,
    "implausibility_cutoff": 3.0,

    # "series" only
    "burn_in_end":       "2012-01-01",
    "fitness_moments": ["gdp_growth_mean", "gdp_growth_std", "unemployment_mean", "unemployment_std",
                        "inflation_std"],
    "fitness_weights": "inverse_noise",
    "observed_data_path": 'analysis/calibration/data/observed_bh.csv',

    # Sobol / SALib settings
    "sobol_calc_second_order": False,
    "sobol_seed":              42,

    # Parameters with S_Ti (or |rho| in fallback mode) below this threshold are candidates to freeze
    "freeze_threshold_sti": 0.05,
}
