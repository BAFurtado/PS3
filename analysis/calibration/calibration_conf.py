# calibration_conf.py

# Parameters to calibrate: [lower_bound, upper_bound]
# Behavioral parameters calibrated on anchor region only.
# All other capitals inherit these values; only structural
# data inputs (A' matrix, ε_s, distributions) vary by region.
CALIBRATION_PARAMETERS = {

    # Production
    "PRODUCTIVITY_MAGNITUDE_DIVISOR": [0.1,   2.0],   # default: 1.0
    "PRODUCTIVITY_EXPONENT":          [0.4,   0.8],   # default: 0.65

    # Pricing
    "MARKUP":                         [0.005, 0.15],  # default: 0.1
    "STICKY_PRICES":                  [0.05,  0.9],   # default: 0.7
    "PRICE_RUGGEDNESS":               [0.05,  0.5],   # default: 0.1

    # Labor market
    "NATURAL_SEPARATION_RATE":        [0.005, 0.04],  # default: 0.01
    "RELEVANCE_UNEMPLOYMENT_SALARIES":[0.5,   5.0],   # default: 1.5
    "LABOR_MARKET":                   [0.3,   1.0],   # default: 0.8
    "PCT_DISTANCE_HIRING":            [0.0,   0.5],   # default: 0.2

    # Inventory
    "INVENTORY_TARGET_RATIO":         [0.0,   0.4],   # default: 0.2

    # Emissions
    "ENVIRONMENTAL_EFFICIENCY_STEP":  [0.90,  0.999], # default: 0.99

}

CALIBRATION_SETTINGS = {

    # Sobol: N * (k + 2) total runs. Use powers of 2.
    # Lower N for scouting, higher for production.
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