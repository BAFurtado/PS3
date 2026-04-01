# conf.py

# Dictionary of parameters to be calibrated: [lower_bound, upper_bound]
CALIBRATION_PARAMETERS = {
    "PRODUCTIVITY_MAGNITUDE_DIVISOR": [0.1, 1.0],
    "MARKUP": [0.1, 0.5]
}

# General calibration settings
CALIBRATION_SETTINGS = {
    "samples": 2,             # Total unique parameter sets to run
    "runs_per_sample": 1,     # Monte Carlo iterations per set
    "target_start_year": 2000,
    "target_end_year": 2010
}