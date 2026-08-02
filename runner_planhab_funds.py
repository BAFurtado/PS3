import argparse
import subprocess
import sys
import pathlib
import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
# Global settings
# ─────────────────────────────────────────────────────────────
#
# MCMV funding-intensity sweep. Any batch produced before 2026-08-01 is void;
# see text/model_defects.md for which and why.
#
# TOTAL_TARGETING_POLICY orders eligible families poorest-first before houses
# are allocated (world/funds.py:update_policy_families). It is the lever the
# CMAP/CMAS (2021) evaluation points at, so it is swept rather than assumed.
#
#   POLICY_MELHORIAS        True | False
#   FUNDS_AVAILABILITY      pessimista | tendencial | otimista
#   TOTAL_TARGETING_POLICY  True | False
#
# = 2 x 3 x 2 = 12 combinations per city; total_jobs = 12 x RUNS.
# POLICY_MCMV stays fixed True -- see main.py's PLANHABFUNDS branch.
#
# Runs on its own dedicated (third) server, independent of runner_planhab.py.

# Use the interpreter this runner was started with, so the child sims cannot
# silently land on a different environment than the one launching them (bare
# `python` is not on PATH outside an activated conda shell).
PYTHON = sys.executable or "python"
MAIN   = "main.py"

SAO_PAULO = "SAO PAULO"

# Applied uniformly to every city. Change these two numbers and every city
# picks them up automatically -- no per-city editing required.
# PLANHABFUNDS now produces 12 combinations per city (see header), so
# total_jobs = 12 x RUNS = 240 per city, 6,240 across the 26 non-SP capitals.
# RUNS stays at 20 rather than dropping to 10: n=20 is what the planhab batch
# already has, the comparison wants matched power, and a third re-run would
# cost far more than the extra wall clock. CPUS_PER_CITY becomes n_jobs in
# joblib's Parallel pool for that city's run.
RUNS           = 20
CPUS_PER_CITY  = 10
MAX_CPU_BUDGET = 10   # this server's total CPU slots for concurrent city runs

# SAO PAULO is an order of magnitude larger than any other ACP. As in
# runner_planhab.py, it can be split off via --only-sp / --exclude-sp if it
# turns out to bottleneck this single-server run; defaults keep it in the
# main batch since there's no second server to dedicate to it here.
SP_RUNS           = 10
SP_CPUS_PER_CITY  = 8
SP_MAX_CPU_BUDGET = 8

MAX_RETRIES    = 1     # extra attempts for failed/crashed cities
TIMEOUT_HOURS  = None  # None = no timeout; set to a number once typical durations are known

LOG_DIR = pathlib.Path("logs/processing_planhabfunds")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Per-city overrides -- only needed for rare exceptions. Empty by default:
# every city uses the uniform (RUNS, CPUS_PER_CITY) / (SP_RUNS,
# SP_CPUS_PER_CITY) settings above.
CITY_CONFIGS = {}

CAPITAIS = [
    SAO_PAULO,
    'RIO DE JANEIRO',
    'BELO HORIZONTE',
    'BRASILIA',
    'CURITIBA',
    'FORTALEZA',
    'GOIANIA',
    'MANAUS',
    'PORTO ALEGRE',
    'RECIFE',
    'SALVADOR',
    'ARACAJU',
    'BELEM',
    'CAMPO GRANDE',
    'CUIABA',
    'FLORIANOPOLIS',
    'JOAO PESSOA',
    'MACEIO',
    'NATAL',
    'SAO LUIS',
    'TERESINA',
    'VITORIA',
    'BOA VISTA',
    'MACAPA',
    'PALMAS',
    'PORTO VELHO',
    'RIO BRANCO',
]


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PLANHABFUNDS (MCMV funding-intensity) sweeps across capitals."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--exclude-sp", action="store_true",
        help="Run every capital except SAO PAULO.",
    )
    group.add_argument(
        "--only-sp", action="store_true",
        help="Run SAO PAULO only.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Weighted CPU semaphore
# ─────────────────────────────────────────────────────────────

class CPUPool:
    """
    Weighted semaphore that tracks total CPU slots.
    Ensures sum of cpus_per_run across simultaneous processes never
    exceeds the configured budget, regardless of how many are queued.
    """
    def __init__(self, total: int):
        self._total = total
        self._available = total
        self._cond = threading.Condition()

    def acquire(self, n: int) -> None:
        with self._cond:
            while self._available < n:
                self._cond.wait()
            self._available -= n

    def release(self, n: int) -> None:
        with self._cond:
            self._available += n
            self._cond.notify_all()

    @property
    def available(self) -> int:
        return self._available

    @property
    def total(self) -> int:
        return self._total


_print_lock = threading.Lock()


def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"{ts}  {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# City runner
# ─────────────────────────────────────────────────────────────

def run_city(city: str, defaults: dict, cpu_pool: CPUPool, attempt: int = 1) -> tuple:
    cfg = CITY_CONFIGS.get(city, defaults)
    runs = cfg['runs']
    cpus = min(cfg['cpus'], cpu_pool.total)  # guard against an impossible acquire()

    cmd = [
        PYTHON, MAIN,
        "-n", str(runs),
        "-c", str(cpus),
        "sensitivity",
        f"PLANHABFUNDS-{city}",
    ]

    safe_name = city.replace(' ', '_')
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix    = f"_attempt{attempt}" if attempt > 1 else ""
    log_file  = LOG_DIR / f"{safe_name}{suffix}_{ts}.log"

    cpu_pool.acquire(cpus)
    t_start = time.monotonic()
    timeout_secs = TIMEOUT_HOURS * 3600 if TIMEOUT_HOURS is not None else None
    _log(f"[{city}] starting  runs={runs} cpus={cpus} attempt={attempt} "
         f"(pool remaining: {cpu_pool.available})")

    try:
        with open(log_file, "w") as lf:
            lf.write(f"CMD:     {' '.join(cmd)}\n")
            lf.write(f"STARTED: {datetime.datetime.now().isoformat()}\n\n")
            lf.flush()

            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=lf,
                timeout=timeout_secs,
            )

        elapsed = time.monotonic() - t_start
        if proc.returncode == 0:
            status = "OK"
        else:
            status = f"FAILED (exit {proc.returncode})"

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t_start
        status = f"TIMEOUT ({TIMEOUT_HOURS}h)"

    except Exception as exc:
        elapsed = time.monotonic() - t_start
        status  = f"CRASHED ({exc})"

    finally:
        cpu_pool.release(cpus)

    return city, status, elapsed, log_file


# ─────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────

def run_batch(cities_and_attempts: list, defaults: dict, cpu_pool: CPUPool) -> dict:
    """
    Submit a batch of (city, attempt) pairs.
    Larger cities are scheduled first so they don't bottleneck at the end
    (only matters if CITY_CONFIGS overrides give some cities more cpus
    than the uniform default).
    Returns {city: status}.
    """
    ordered = sorted(
        cities_and_attempts,
        key=lambda ca: CITY_CONFIGS.get(ca[0], defaults)['cpus'],
        reverse=True,
    )

    batch_results: dict = {}
    # max_workers >= len(ordered) so every city can block on the semaphore
    # concurrently; actual parallelism is capped by cpu_pool
    with ThreadPoolExecutor(max_workers=len(ordered)) as executor:
        futures = {
            executor.submit(run_city, city, defaults, cpu_pool, attempt): city
            for city, attempt in ordered
        }
        for future in as_completed(futures):
            city, status, elapsed, log_file = future.result()
            mins = elapsed / 60
            _log(f"[{city}] {status}  ({mins:.1f} min)  → {log_file.name}")
            batch_results[city] = status

    return batch_results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.only_sp:
        cities   = [SAO_PAULO]
        defaults = {'runs': SP_RUNS, 'cpus': SP_CPUS_PER_CITY}
        budget   = SP_MAX_CPU_BUDGET
        label    = "SAO PAULO only"
    elif args.exclude_sp:
        cities   = [c for c in CAPITAIS if c != SAO_PAULO]
        defaults = {'runs': RUNS, 'cpus': CPUS_PER_CITY}
        budget   = MAX_CPU_BUDGET
        label    = f"{len(cities)} cities (excluding SAO PAULO)"
    else:
        cities   = list(CAPITAIS)
        defaults = {'runs': RUNS, 'cpus': CPUS_PER_CITY}
        budget   = MAX_CPU_BUDGET
        label    = f"{len(cities)} cities (including SAO PAULO)"

    cpu_pool = CPUPool(budget)

    timeout_label = f"{TIMEOUT_HOURS}h" if TIMEOUT_HOURS is not None else "none"
    _log(f"PlanHabFunds runner: {label}  CPU budget={budget}  "
         f"runs={defaults['runs']} cpus/city={defaults['cpus']}  "
         f"timeout={timeout_label}  retries={MAX_RETRIES}")

    results = run_batch([(city, 1) for city in cities], defaults, cpu_pool)

    failed = [city for city, status in results.items() if status != "OK"]
    if failed and MAX_RETRIES > 0:
        sep = "─" * 55
        _log(f"\n{sep}\nRetrying {len(failed)} cities: {failed}\n{sep}")
        retry_results = run_batch([(city, 2) for city in failed], defaults, cpu_pool)
        results.update(retry_results)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print("SUMMARY")
    print(f"{'═' * 55}")
    ok_count = 0
    for city in cities:
        status = results.get(city, "NOT RUN")
        icon   = "v" if status == "OK" else "x"
        print(f"  [{icon}] {city:<22s}  {status}")
        if status == "OK":
            ok_count += 1
    print(f"{'─' * 55}")
    print(f"  {ok_count}/{len(cities)} cities completed successfully.")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()