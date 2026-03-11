import subprocess
import itertools
import pathlib
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PYTHON = "python"
MAIN = "main.py"

RUNS = 3
CPUS_PER_RUN = 3         # internal parallelism inside main.py
MAX_PARALLEL_RUNS = 3     # number of simultaneous OS processes

LOG_DIR = pathlib.Path("logs/processing_acps")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CAPITAIS = [
    'ARACAJU',
    'BELEM',
    'BELO HORIZONTE',
    # 'BOA VISTA',
    'BRASILIA',
    'CAMPO GRANDE',
    'CUIABA',
    'CURITIBA',
    'FORTALEZA',
    'FLORIANOPOLIS',
    'GOIANIA',
    'JOAO PESSOA',
    'MACAPA',
    'MACEIO',
    'MANAUS',
    'NATAL',
    'PALMAS',
    'PORTO ALEGRE',
    'PORTO VELHO',
    'RECIFE',
    'RIO BRANCO',
    'SAO LUIS',
    'SALVADOR',
    'TERESINA',
    'VITORIA'
]


# Optional: policy / interest slicing at runner level
POLICY_MCMV = [True, False]
POLICY_MELHORIAS = [True, False]
INTEREST = ['baixa', 'media', 'alta']

# ---------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------


def build_command(city):
    """
    One atomic PLANHAB run = one city.
    All policy combinations stay inside sensitivity().
    """
    return [
        PYTHON, MAIN,
        "-n", str(RUNS),
        "-c", str(CPUS_PER_RUN),
        "sensitivity",
        # f"PLANHAB-{city}",
        f"PROCESSING_ACP-{city}"
    ]

# ---------------------------------------------------------------------
# Execution wrapper
# ---------------------------------------------------------------------


def run_city(city):
    cmd = build_command(city)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{city}_{timestamp}.log"

    try:
        with open(log_file, "w") as log:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,  # discard stdout
                stderr=log,  # keep errors only
                check=True
            )
        return city, "OK"

    except subprocess.CalledProcessError as e:
        return city, f"FAILED (exit code {e.returncode})"

    except Exception as e:
        return city, f"CRASHED ({e})"

# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------


def main():
    results = []

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_RUNS) as executor:
        futures = {
            executor.submit(run_city, city): city
            for city in CAPITAIS
        }

        for future in as_completed(futures):
            city, status = future.result()
            print(f"[{city}] {status}")
            results.append((city, status))

    print("\nSummary")
    print("-" * 40)
    for city, status in results:
        print(f"{city:15s} {status}")


if __name__ == "__main__":
    main()
