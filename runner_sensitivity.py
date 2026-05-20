import json
import subprocess
import pathlib
import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PYTHON = "python"
MAIN = "main.py"

# 5 credit parameters only — re-run after tabela Price/SAC commit (2026-05-20).
# Previous wave21 sensitivity used old loan math; these results are now stale.
# CPUS=6 reserved for sensitivity; runner_planhab.py uses the remaining cores.
# With 6 CPUs: 4-pt params → 2 batches of 6 jobs ≈ 40 min each; 3-pt → same.
# 5 params × ~40 min ≈ 3.5 h total — fits comfortably alongside capitals run.
CITY = "GOIANIA"
RUNS = 3
CPUS = 6

LOG_DIR = pathlib.Path("logs/sensitivity_wave22_credit")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CITY_PARAMS_PATH = LOG_DIR / "city_params.json"
with open(CITY_PARAMS_PATH, "w") as _f:
    json.dump({"PROCESSING_ACPS": [CITY]}, _f)

# ---------------------------------------------------------------------
# OAT Parameter Ranges
# Format: PARAM: (start, end, n_points)
# Ranges are centred on / include the current default.
# With RUNS=3 and CPUS=6: 4-pt params → 12 jobs, 2 sequential batches of 6.
# ---------------------------------------------------------------------

PARAM_RANGES = {

    # ── Subsidised credit access — income eligibility ───────────────
    # Quantile of the income distribution below which families qualify.
    # Exposed as scalar aliases of INCOME_MODALIDADES['fgts'/'sbpe'].
    # current fgts=0.65: bottom 65% eligible for FGTS.
    "FGTS_INCOME_QUANTILE":      (0.50, 0.80, 4),
    # current sbpe=0.85: families in 65th–85th percentile eligible for SBPE.
    "SBPE_INCOME_QUANTILE":      (0.70, 0.95, 4),

    # ── Subsidised credit access — loan size ────────────────────────
    # current=0.95 (FGTS allows effectively 0–5% down payment).
    # FGTS now uses Tabela Price (correct first-payment formula) — results
    # from wave21 used the old SAC-everywhere math and are stale for these.
    "MAX_LOAN_TO_VALUE_FGTS":    (0.85, 1.00, 4),
    # current=0.90
    "MAX_LOAN_TO_VALUE_SBPE":    (0.80, 0.95, 4),

    # ── Housing purchase entry gate ─────────────────────────────────
    # current=0.20; lower values allow families with less equity to enter.
    "MIN_DOWN_PAYMENT_FRACTION": (0.05, 0.30, 3),
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