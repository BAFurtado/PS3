import json
import subprocess
import pathlib
import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PYTHON = "python"
MAIN = "main.py"

# Goiania: 5/5 targets in 2011-2021, mid-sized, shows long-run Gini drift.
# ~47 min/param with -c 18 (all points × runs fit in parallel).
# 12 params × 47 min ≈ 9.5 h total.
CITY = "GOIANIA"
RUNS = 3
CPUS = 18

LOG_DIR = pathlib.Path("logs/sensitivity_wave21")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CITY_PARAMS_PATH = LOG_DIR / "city_params.json"
with open(CITY_PARAMS_PATH, "w") as _f:
    json.dump({"PROCESSING_ACPS": [CITY]}, _f)

# ---------------------------------------------------------------------
# OAT Parameter Ranges
# Format: PARAM: (start, end, n_points)
# Ranges are centred on / include the current default.
# With RUNS=3 and CPUS=18, all (n_points × 3) jobs run in parallel
# when n_points × 3 ≤ 18, keeping wall time ≈ 1 simulation per param.
# ---------------------------------------------------------------------

PARAM_RANGES = {

    # ── Spatial price heterogeneity ─────────────────────────────────
    # price *= (1 + NE × normalised_neighbourhood_median_income)
    # current=0.5; was 2.0 (caused zero_cons explosion); zero_cons now fixed,
    # so we explore the full range to find the Gini/access trade-off.
    # 5 points: 18 CPUs fit 5×3=15 jobs simultaneously.
    "NEIGHBORHOOD_EFFECT":          (0.0,  2.0,  5),

    # ── Subsidised credit access — income eligibility ───────────────
    # Quantile of the income distribution below which families qualify.
    # Exposed as scalar aliases of INCOME_MODALIDADES['fgts'/'sbpe'].
    # current fgts=0.65: bottom 65% eligible for FGTS.
    "FGTS_INCOME_QUANTILE":         (0.50, 0.80, 4),
    # current sbpe=0.85: families in 65th–85th percentile eligible for SBPE.
    "SBPE_INCOME_QUANTILE":         (0.70, 0.95, 4),

    # ── Subsidised credit access — loan size ────────────────────────
    # current=0.95 (FGTS allows effectively 0–5% down payment).
    "MAX_LOAN_TO_VALUE_FGTS":       (0.85, 1.00, 4),
    # current=0.90
    "MAX_LOAN_TO_VALUE_SBPE":       (0.80, 0.95, 4),

    # ── Housing purchase entry gate ─────────────────────────────────
    # current=0.20; lower values allow families with less equity to enter.
    "MIN_DOWN_PAYMENT_FRACTION":    (0.05, 0.30, 3),
    # current=5.0; penalises investment-motivated buyers when rates > rental yield.
    "HOUSING_FINANCIAL_WEIGHT":     (1.0,  10.0, 3),

    # ── Construction supply ─────────────────────────────────────────
    # Exponential vacancy suppression: P(skip) = 1 − exp(−vacancy × BVS).
    # current=10; lower → more building at any given vacancy.
    "BUILD_VACANCY_SENSITIVITY":    (5,    20,   4),
    # current=9; bridges scale gap between firm labour output and m² built.
    "HOUSE_PRODUCTION_ADEQUACY":    (5,    15,   3),
    # current=12; months over which construction revenue is recognised.
    "CONSTRUCTION_ACC_CASH_FLOW":   (1,    24,   3),

    # ── Vacancy–price equilibrium ───────────────────────────────────
    # Vacancy at which listed prices sit at base level.
    # current=0.08; 3 cities (Brasília, Campo Grande, Palmas) run just below.
    "VACANCY_PRICE_REFERENCE":      (0.04, 0.12, 3),

    # ── Quality-of-life dynamics ────────────────────────────────────
    # Elasticity of QLI to per-capita population pressure.
    # current=0.3; 0 = no crowding effect; 1 = pure per-capita sharing.
    "QLI_POP_ELASTICITY":           (0.0,  0.6,  3),
}

# ---------------------------------------------------------------------
# Execution loop (sequential across params; parallel runs inside each)
# ---------------------------------------------------------------------


def main():

    total = len(PARAM_RANGES)
    count = 1

    for param, (start, end, steps) in PARAM_RANGES.items():

        print(f"\n[{count}/{total}] {param}  ({start} → {end}, {steps} points, city={CITY})")

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
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log, check=True)
            print(f"  ✓ completed")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED (exit code {e.returncode})")
        except Exception as e:
            print(f"  ✗ CRASHED ({e})")

        count += 1

    print(f"\nAll {total} sensitivity runs done. City: {CITY}")


if __name__ == "__main__":
    main()