import subprocess
import pathlib
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PYTHON = "python"
MAIN = "main.py"

RUNS = 5
CPUS_PER_RUN = 6          # ajuste conforme servidor
MAX_PARALLEL_RUNS = 3     # processos simultâneos

LOG_DIR = pathlib.Path("logs/planhab_oat")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# OAT Parameter Ranges
# Formato: PARAM:inicio:fim:numero_de_pontos
# ---------------------------------------------------------------------

PARAM_RANGES = {
    "OFFER_SIZE_ON_PRICE": (1, 3, 3),
    "ON_MARKET_DECAY_FACTOR": (-0.06, 0, 3),
    "MAX_OFFER_DISCOUNT": (0.2, 0.8, 3),
    "NEIGHBORHOOD_EFFECT": (0, 2, 3),
    "PERCENTAGE_ENTERING_ESTATE_MARKET": (0.03, 0.07, 3),
    "EXPECTED_LICENSES_PER_REGION": (0, 2, 3),
    "CONSTRUCTION_FIRM_MARKUP_MULTIPLIER": (3, 9, 3),
    "LOT_COST": (0.05, 0.30, 3),
    "MARKUP": (0.05, 0.2, 3),
    "MAX_LOAN_TO_VALUE": (0.4, 1, 3),
    "LOAN_PAYMENT_TO_PERMANENT_INCOME": (0.10, 0.50, 3),
    "PRODUCTIVITY_EXPONENT": (0.35, 0.65, 3),
    "RELEVANCE_UNEMPLOYMENT_SALARIES": (1, 5, 3),
    "POLICY_MCMV_PERCENTAGE": (0.1, .3, 3),
    "CONSTRUCTION_ACC_CASH_FLOW": (12, 36, 3),

}

# ---------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------

def build_command(param_string):
    return [
        PYTHON, MAIN,
        "-n", str(RUNS),
        "-c", str(CPUS_PER_RUN),
        "sensitivity",
        param_string
    ]

# ---------------------------------------------------------------------
# Execution wrapper
# ---------------------------------------------------------------------

def run_simulation(label, param_string):

    cmd = build_command(param_string)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{label}_{timestamp}.log"

    try:
        with open(log_file, "w") as log:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=log,
                check=True
            )
        return label, "OK"

    except subprocess.CalledProcessError as e:
        return label, f"FAILED (exit code {e.returncode})"

    except Exception as e:
        return label, f"CRASHED ({e})"

# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def main():

    jobs = []

    for param, (start, end, steps) in PARAM_RANGES.items():
        param_string = f"{param}:{start}:{end}:{steps}"
        label = param
        jobs.append((label, param_string))

    results = []

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_RUNS) as executor:

        futures = {
            executor.submit(run_simulation, label, param_string): label
            for label, param_string in jobs
        }

        for future in as_completed(futures):
            label, status = future.result()
            print(f"[{label}] {status}")
            results.append((label, status))

    print("\nSummary")
    print("-" * 50)
    for label, status in results:
        print(f"{label:40s} {status}")

if __name__ == "__main__":
    main()