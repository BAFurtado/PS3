import json
import subprocess
import pathlib
import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PYTHON = "python"
MAIN = "main.py"

CITY = "GOIANIA"   # canonical sensitivity city (~47 min/run; 18 params × 47 min ≈ 14 h)
RUNS = 3
CPUS = 18

LOG_DIR = pathlib.Path("logs/planhab_oat")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Write city override once so every subprocess picks it up via -p
CITY_PARAMS_PATH = LOG_DIR / "city_params.json"
with open(CITY_PARAMS_PATH, "w") as _f:
    json.dump({"PROCESSING_ACPS": [CITY]}, _f)

# ---------------------------------------------------------------------
# OAT Parameter Ranges
# Format: PARAM: (start, end, n_points)
# All ranges are centred on / include the current default value.
# ---------------------------------------------------------------------

PARAM_RANGES = {
    # ── Housing market pricing ──────────────────────────────────────
    # current=3; upper bound raised so sweep covers the current value mid-range
    "OFFER_SIZE_ON_PRICE":                   (1,      5,      3),
    # current=-0.02; range covers faster decay through no decay
    "ON_MARKET_DECAY_FACTOR":                (-0.04,  0,      3),
    # current=0.65; unchanged — centred inside range
    "MAX_OFFER_DISCOUNT":                    (0.4,    0.8,    3),
    # current=2; OLD range (0,1) did not cover the current value — fixed
    "NEIGHBORHOOD_EFFECT":                   (0,      4,      3),

    # ── Housing market entry ────────────────────────────────────────
    # current=0.005; OLD range (0.03, 0.07) was 6–14× the current value — fixed
    "PERCENTAGE_ENTERING_ESTATE_MARKET":     (0.002,  0.015,  3),
    # NEW param (down-payment gate); current=0.20
    "MIN_DOWN_PAYMENT_FRACTION":             (0.10,   0.30,   3),
    # NEW param (investment-vs-rent deterrence); current=3.0
    "HOUSING_FINANCIAL_WEIGHT":              (1.0,    6.0,    3),

    # ── Rental market ───────────────────────────────────────────────
    # NEW in sensitivity; recently changed 0.25→0.40
    "INITIAL_RENTAL_SHARE":                  (0.25,   0.55,   3),
    # NEW in sensitivity; recently changed 0.0015→0.003
    "INITIAL_RENTAL_PRICE":                  (0.0015, 0.005,  3),

    # ── Construction supply ─────────────────────────────────────────
    # NEW (user-requested); current=12 — key vacancy throttle
    "BUILD_VACANCY_SENSITIVITY":             (6,      18,     3),
    # current=1; OLD range (0,1) had current at the edge — extended to 3
    "EXPECTED_LICENSES_PER_REGION":          (0,      3,      3),
    # current=3; OLD range (4,8) started above the current value — fixed
    "CONSTRUCTION_FIRM_MARKUP_MULTIPLIER":   (1,      8,      3),
    # current=0.15; unchanged
    "LOT_COST":                              (0.10,   0.20,   3),
    # NEW param (production scale bridge); current=15
    "HOUSE_PRODUCTION_ADEQUACY":             (8,      25,     3),
    # current=12; range tests the effect of shorter cash-flow smoothing
    "CONSTRUCTION_ACC_CASH_FLOW":            (1,      12,     3),

    # ── Credit ──────────────────────────────────────────────────────
    # current=0.8; unchanged
    "MAX_LOAN_TO_VALUE":                     (0.7,    0.9,    3),
    # current=0.35; slightly wider than old (0.30, 0.40)
    "LOAN_PAYMENT_TO_PERMANENT_INCOME":      (0.25,   0.45,   3),

    # ── Production function ─────────────────────────────────────────
    # current=0.8; OLD range (0.4, 0.6) did not cover the current value — fixed
    "PRODUCTIVITY_EXPONENT":                 (0.5,    1.0,    3),
}

# ---------------------------------------------------------------------
# Execution Loop (sequential parameters, parallel runs inside each)
# ---------------------------------------------------------------------


def main():

    total = len(PARAM_RANGES)
    count = 1

    for param, (start, end, steps) in PARAM_RANGES.items():

        print(f"\n[{count}/{total}] Running sensitivity for {param}  "
              f"({start} → {end}, {steps} points, city={CITY})")

        param_string = f"{param}:{start}:{end}:{steps}"

        cmd = [
            PYTHON, MAIN,
            "-p", str(CITY_PARAMS_PATH),
            "-n", str(RUNS),
            "-c", str(CPUS),
            "sensitivity",
            param_string,
        ]

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"{param}_{timestamp}.log"

        try:
            with open(log_file, "w") as log:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    check=True,
                )

            print(f"✓ {param} completed")

        except subprocess.CalledProcessError as e:
            print(f"✗ {param} FAILED (exit code {e.returncode})")

        except Exception as e:
            print(f"✗ {param} CRASHED ({e})")

        count += 1

    print(f"\nAll {total} sensitivity runs completed. City: {CITY}")


if __name__ == "__main__":
    main()
