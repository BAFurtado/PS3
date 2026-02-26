import subprocess
import pathlib
import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PYTHON = "python"
MAIN = "main.py"

RUNS = 5
CPUS = 18   # máximo de cores que você quer usar no servidor

LOG_DIR = pathlib.Path("logs/planhab_oat")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# OAT Parameter Ranges
# Formato: PARAM:inicio:fim:numero_de_pontos
# ---------------------------------------------------------------------

PARAM_RANGES = {
    "OFFER_SIZE_ON_PRICE": (1, 3, 3),
    "ON_MARKET_DECAY_FACTOR": (-0.04, 0, 3),
    "MAX_OFFER_DISCOUNT": (0.4, 0.8, 3),
    "NEIGHBORHOOD_EFFECT": (0, 1, 3),
    "PERCENTAGE_ENTERING_ESTATE_MARKET": (0.03, 0.07, 3),
    "EXPECTED_LICENSES_PER_REGION": (0, 1, 3),
    "CONSTRUCTION_FIRM_MARKUP_MULTIPLIER": (4, 8, 3),
    "LOT_COST": (0.10, 0.20, 3),
    "CONSTRUCTION_ACC_CASH_FLOW": (1, 12, 3),
    "MAX_LOAN_TO_VALUE": (0.7, 0.9, 3),
    "LOAN_PAYMENT_TO_PERMANENT_INCOME": (0.30, 0.40, 3),
    "PRODUCTIVITY_EXPONENT": (0.4, 0.6, 3),
}

# ---------------------------------------------------------------------
# Execution Loop (sequencial)
# ---------------------------------------------------------------------

def main():

    total = len(PARAM_RANGES)
    count = 1

    for param, (start, end, steps) in PARAM_RANGES.items():

        print(f"\n[{count}/{total}] Running sensitivity for {param}")

        param_string = f"{param}:{start}:{end}:{steps}"

        cmd = [
            PYTHON, MAIN,
            "-n", str(RUNS),
            "-c", str(CPUS),
            "sensitivity",
            param_string
        ]

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"{param}_{timestamp}.log"

        try:
            with open(log_file, "w") as log:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    check=True
                )

            print(f"✓ {param} completed")

        except subprocess.CalledProcessError as e:
            print(f"✗ {param} FAILED (exit code {e.returncode})")

        except Exception as e:
            print(f"✗ {param} CRASHED ({e})")

        count += 1

    print("\nAll sensitivity runs completed.")


if __name__ == "__main__":
    main()