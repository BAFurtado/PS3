import json
import subprocess
import pathlib
import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PYTHON = "python"
MAIN = "main.py"

# Wave30 OAT — 4 housing/market params centred on current wave30 defaults.
# Runs in parallel with the wave30 capitals run (6 CPUs reserved here).
# 4 params × 4 points × 3 runs = 48 jobs → 8 batches of 6 → ~4-5 h total.
CITY = "GOIANIA"
RUNS = 3
CPUS = 6

LOG_DIR = pathlib.Path("logs/sensitivity_wave30_housing")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CITY_PARAMS_PATH = LOG_DIR / "city_params.json"
with open(CITY_PARAMS_PATH, "w") as _f:
    json.dump({"PROCESSING_ACPS": [CITY]}, _f)

# ---------------------------------------------------------------------
# OAT Parameter Ranges
# Format: PARAM: (start, end, n_points)  →  np.linspace(start, end, n_points)
#
# HFW  25→100:  [25, 50, 75, 100]  — primary unemployment/vacancy lever
# HPA   6→12:  [ 6,  8, 10,  12]  — production volume control
# BVS   6→15:  [ 6,  9, 12,  15]  — construction response to vacancy
# NEIGH 0→0.6: [ 0, 0.2, 0.4, 0.6] — zero-consumption amplifier
#
# Wave30 defaults: HFW=60  HPA=8  BVS=9  NEIGH=0.2
# All four current values fall inside their respective ranges.
# Wave12 OAT is outdated (HFW range was 1–10; major structural changes since).
# These results will supersede wave12 correlations.
# ---------------------------------------------------------------------

# Categorical params — use the PARAM*val1+val2+... syntax (already handled by main.py).
# real  = BC series 433 real estate financing rate (~3%/yr mean, 2010–2022)
# media = blended average scenario (~9.2%/yr, current default)
# nominal = raw SELIC-based (~8.0%/yr mean, 2010–2022)
# sbpe/fgts rates are identical across all three (regulated, SELIC-independent).
CATEGORICAL_PARAMS = {
    "INTEREST": ["real", "media", "nominal"],
}

PARAM_RANGES = {

    # ── Housing purchase decision — opportunity-cost weight ─────────
    # Wave28: 100 → unemployment OK, vacancy 4% (too low)
    # Wave29: 25  → unemployment 2.2% (collapsed), vacancy 6.4%
    # Wave30: 60  → target unemployment 6-8%, vacancy 7-9%
    # OAT r (wave12, directional): Vacancy +0.93  Unemp +0.76  ZeroCons -0.92
    "HOUSING_FINANCIAL_WEIGHT":  (25, 100, 4),

    # ── Construction output per license ────────────────────────────
    # Wave29: 10 → production 4.6/1000 (target 2-4, too high)
    # Wave30:  8 → target ~3.0-3.5/1000
    # OAT r (wave12, directional): Vacancy +0.98  Unemp -0.64  ZeroCons -0.93
    "HOUSE_PRODUCTION_ADEQUACY": (6, 12, 4),

    # ── Construction sensitivity to vacancy level ───────────────────
    # Wave28: 13 → Wave29: 9 → vacancy improved but production overshot
    # Wave30:  9 held — sweep clarifies balance point with new HPA=8
    # OAT r (wave12, directional): Vacancy -0.56  Gini +0.53  LoanAppr -0.58
    "BUILD_VACANCY_SENSITIVITY": (6, 15, 4),

    # ── Neighbourhood feedback amplifier ───────────────────────────
    # OAT r=+1.00 with zero consumption — strongest single lever in matrix
    # Current 0.2 (user chose as compromise); wave12 tested 0-2.0
    # OAT r (wave12, directional): ZeroCons +1.00  Vacancy +0.72  Gini -0.57
    "NEIGHBORHOOD_EFFECT":       (0.0, 0.6, 4),
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
        print(f"  ✓ completed")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ FAILED (exit code {e.returncode})")
    except Exception as e:
        print(f"  ✗ CRASHED ({e})")


def main():

    total = len(PARAM_RANGES) + len(CATEGORICAL_PARAMS)
    count = 1

    for param, (start, end, steps) in PARAM_RANGES.items():
        param_string = f"{param}:{start}:{end}:{steps}"
        label = f"{param}_{start}-{end}"
        run_sensitivity(param_string, label, count, total)
        count += 1

    for param, values in CATEGORICAL_PARAMS.items():
        param_string = f"{param}*{'+'.join(values)}"
        label = f"{param}_{'_'.join(values)}"
        run_sensitivity(param_string, label, count, total)
        count += 1

    print(f"\nAll {total} sensitivity sweeps done.  City: {CITY}")
    print(f"Logs → {LOG_DIR}")
    print("Next: python analysis/sensitivity_oat.py output")


if __name__ == "__main__":
    main()