import json
import subprocess
import pathlib
import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PYTHON = "python"
MAIN = "main.py"

# Wave33 OAT — re-evaluation after the two construction-dynamics mechanisms
# added this session:
#   1. labor_signals(): Construction-specific hire/fire (backlog/oversupply
#      driven, not goods-market increase_production/profit).
#   2. plan_house() vacancy_factor (Prong B): expected_price is now scaled by
#      the same vacancy discount/premium House.update_price() applies when
#      listing, so a saturated market directly suppresses profitability,
#      not just the build_sensitivity skip-probability.
#
# Both mechanisms are new and BVS/HPA/MHS/HFW/LOT_COST were last tuned
# (wave29-32) without them. Goal: check whether prior settings still hold,
# and whether BVS can come down now that vacancy_factor adds a second brake.
#
# 6 params x 4 points x 3 runs = 72 jobs -> 12 batches of 6 -> ~5-6h total.
CITY = "GOIANIA"
RUNS = 3
CPUS = 6

LOG_DIR = pathlib.Path("logs/sensitivity_wave33_construction")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CITY_PARAMS_PATH = LOG_DIR / "city_params.json"
with open(CITY_PARAMS_PATH, "w") as _f:
    json.dump({"PROCESSING_ACPS": [CITY]}, _f)

# ---------------------------------------------------------------------
# OAT Parameter Ranges
# Format: PARAM: (start, end, n_points) -> np.linspace(start, end, n_points)
#
# BVS    5->13: current=9.  vacancy_factor is now a second vacancy brake on
#               plan_house (Prong B); check whether BVS's skip-probability
#               can be relaxed without vacancy re-overshooting.
# OSP    1->5:  current=3.  OFFER_SIZE_ON_PRICE sets the SLOPE of
#               vacancy_factor in BOTH update_price() and (now) plan_house's
#               profitability -- the core dial of Prong B.
# MHS   20->50: current=36. MAX_HOUSE_STOCK now gates BOTH plan_house's early
#               skip and labor_signals' oversupplied/fire decision.
# HPA    8->16: current=12. Production-volume control; interacts with
#               labor_signals' capacity_short (backlog vs total_quantity).
# HFW   30->90: current=60. Demand-side housing-entry lever; re-check given
#               the new supply-side dynamics.
# LOT   0.10->0.20: current=0.15. Cost-side counterpart in the same profit
#               formula Prong B modifies on the revenue side.
# ---------------------------------------------------------------------

PARAM_RANGES = {
    "BUILD_VACANCY_SENSITIVITY":  (5, 13, 4),
    "OFFER_SIZE_ON_PRICE":        (1, 5, 4),
    "MAX_HOUSE_STOCK":            (20, 50, 4),
    "HOUSE_PRODUCTION_ADEQUACY":  (8, 16, 4),
    "HOUSING_FINANCIAL_WEIGHT":   (30, 90, 4),
    "LOT_COST":                   (0.10, 0.20, 4),
}

# ---------------------------------------------------------------------
# Execution loop (sequential across params; parallel runs inside each)
# ---------------------------------------------------------------------


def run_sensitivity(param_string, label, count, total):
    print(f"\n[{count}/{total}] {label}  (city={CITY})")
    cmd = [
        PYTHON, MAIN,
        "-p", str(CITY_PARAMS_PATH),
        "-n", str(RUNS),
        "-c", str(CPUS),
        "sensitivity",
        param_string,
    ]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{label}_{timestamp}.log"
    try:
        with open(log_file, "w") as log:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log, check=True)
        print(f"  done")
    except subprocess.CalledProcessError as e:
        print(f"  FAILED (exit code {e.returncode})")
    except Exception as e:
        print(f"  CRASHED ({e})")


def main():
    total = len(PARAM_RANGES)
    count = 1

    for param, (start, end, steps) in PARAM_RANGES.items():
        param_string = f"{param}:{start}:{end}:{steps}"
        label = f"{param}_{start}-{end}"
        run_sensitivity(param_string, label, count, total)
        count += 1

    print(f"\nAll {total} sensitivity sweeps done.  City: {CITY}")
    print(f"Logs -> {LOG_DIR}")
    print("Next: python analysis/sensitivity_oat.py output")


if __name__ == "__main__":
    main()
