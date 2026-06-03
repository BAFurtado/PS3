# calibration_conf.py

# Parameters to calibrate: [lower_bound, upper_bound]
# Behavioral parameters calibrated on anchor region only.
# All other capitals inherit these values; only structural
# data inputs (A' matrix, ε_s, distributions) vary by region.
CALIBRATION_PARAMETERS = {

    "PRODUCTIVITY_MAGNITUDE_DIVISOR": [0.1, 2.0],   # paper 1: 8.25
    #"ECO_INVESTMENT_LAMBDA":          [5.0, 20.0],   # paper 1: 10

    "MARKUP":                         [0.005, 0.1],  # paper 1: 0.1
    "RELEVANCE_UNEMPLOYMENT_SALARIES":[1.0,   5.0],  # paper 1: 3.5

    "STICKY_PRICES":                  [0.05,   0.4],  # paper 1: 0.5
    #"REGIONAL_FREIGHT_COST":          [0.1,   0.5],  # paper 1: 0.3

}

CALIBRATION_SETTINGS = {

    # Sobol: N * (k + 2) total runs.
    # k=7 (full set) → 128 * 9 = 1152 runs.
    # Use samples=64 for scouting, 128 for production.
    "samples":        128,
    "runs_per_sample": 1,   # min 5 for stable stochastic estimates

    # Burn-in excluded from fitness; moments computed over [burn_in_end, target_end_year]
    "burn_in_end":       "2012-01-01",
    "target_start_year": "2010-01-01",
    "target_end_year":   "2015-01-01",

    # Anchor region for calibration
    "calibration_region": "BELO HORIZONTE",

    # Sobol / SALib settings
    "sobol_calc_second_order": False,
    "sobol_seed":              42,

    "observed_data_path": 'analysis/calibration/data/observed_bh.csv',

    # Parameters with S_Ti (or |rho| in fallback mode) below this threshold are candidates to freeze
    "freeze_threshold_sti": 0.05,
}