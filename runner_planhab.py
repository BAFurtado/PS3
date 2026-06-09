import subprocess
import pathlib
import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
# Global settings
# ─────────────────────────────────────────────────────────────

PYTHON         = "python"
MAIN           = "main.py"

MAX_CPU_BUDGET = 12    # total CPU slots across all parallel city runs (server has 12)
MAX_RETRIES    = 1     # extra attempts for failed/crashed cities
TIMEOUT_HOURS  = None  # None = no timeout; set to a number once typical durations are known

LOG_DIR = pathlib.Path("logs/processing_acps")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Per-city configuration  (runs, cpus)
# Cities not listed here fall back to DEFAULTS.
# ─────────────────────────────────────────────────────────────

# PLANHAB produces 2 (POLICY_MELHORIAS) × 3 (FUNDS_AVAILABILITY) = 6 combinations per city.
# With n runs each: total_jobs = 6 × n.  cpus becomes n_jobs in joblib's Parallel pool.
# Clean divisors of 6×3=18: 1, 2, 3, 6, 9, 18.  Use 6/3/2 to keep workers fully busy.

DEFAULTS = {'runs': 3, 'cpus': 3}

CITY_CONFIGS = {
    # ── Very large metros (multi-day wall time expected) ──────
    'SAO PAULO':      {'runs': 3, 'cpus': 6},  # 18 jobs / 6 workers = 3 full rounds
    'RIO DE JANEIRO': {'runs': 3, 'cpus': 6},
    # ── Large metros ─────────────────────────────────────────
    # cpus=3 → 6 full rounds; 3 cities fit simultaneously in the 10-CPU budget (3+3+3=9)
    'BELO HORIZONTE': {'runs': 3, 'cpus': 3},
    'BRASILIA':       {'runs': 3, 'cpus': 3},
    'CURITIBA':       {'runs': 3, 'cpus': 3},
    'FORTALEZA':      {'runs': 3, 'cpus': 3},
    'GOIANIA':        {'runs': 3, 'cpus': 3},
    'MANAUS':         {'runs': 3, 'cpus': 3},
    'PORTO ALEGRE':   {'runs': 3, 'cpus': 3},
    'RECIFE':         {'runs': 3, 'cpus': 3},
    'SALVADOR':       {'runs': 3, 'cpus': 3},
    # ── Small/remote capitals ─────────────────────────────────
    # cpus=2 → 9 full rounds; up to 5 cities fit simultaneously (2×5=10)
    'BOA VISTA':      {'runs': 3, 'cpus': 2},
    'MACAPA':         {'runs': 3, 'cpus': 2},
    'PALMAS':         {'runs': 3, 'cpus': 2},
    'PORTO VELHO':    {'runs': 3, 'cpus': 2},
    'RIO BRANCO':     {'runs': 3, 'cpus': 2},
}

# Cities still missing complete runs as of stats36 (2026-06-09).
# 10 cities had zero planhab runs; SAO PAULO needs pessimista/True completed
# (1 of 3 runs exists). Running full PLANHAB for all 11 is safe — duplicates
# are harmless and the aggregation script picks any 3 complete runs per combo.
CAPITAIS = [
    'BOA VISTA',
    'CUIABA',
    'FORTALEZA',
    'MACAPA',
    'MACEIO',
    'NATAL',
    'PALMAS',
    'SAO LUIS',
    'SALVADOR',
    'TERESINA',
]


# ─────────────────────────────────────────────────────────────
# Weighted CPU semaphore
# ─────────────────────────────────────────────────────────────

class CPUPool:
    """
    Weighted semaphore that tracks total CPU slots.
    Ensures sum of cpus_per_run across simultaneous processes never
    exceeds MAX_CPU_BUDGET, regardless of how many are queued.
    """
    def __init__(self, total: int):
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


_cpu_pool = CPUPool(MAX_CPU_BUDGET)
_print_lock = threading.Lock()


def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"{ts}  {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# City runner
# ─────────────────────────────────────────────────────────────

def run_city(city: str, attempt: int = 1) -> tuple:
    cfg = CITY_CONFIGS.get(city, DEFAULTS)
    runs = cfg['runs']
    cpus = min(cfg['cpus'], MAX_CPU_BUDGET)  # guard against impossible acquire

    cmd = [
        PYTHON, MAIN,
        "-n", str(runs),
        "-c", str(cpus),
        "sensitivity",
        f"PLANHAB-{city}",
    ]

    safe_name = city.replace(' ', '_')
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix    = f"_attempt{attempt}" if attempt > 1 else ""
    log_file  = LOG_DIR / f"{safe_name}{suffix}_{ts}.log"

    _cpu_pool.acquire(cpus)
    t_start = time.monotonic()
    timeout_secs = TIMEOUT_HOURS * 3600 if TIMEOUT_HOURS is not None else None
    _log(f"[{city}] starting  runs={runs} cpus={cpus} attempt={attempt} "
         f"(pool remaining: {_cpu_pool.available})")

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
        _cpu_pool.release(cpus)

    return city, status, elapsed, log_file


# ─────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────

def run_batch(cities_and_attempts: list) -> dict:
    """
    Submit a batch of (city, attempt) pairs.
    Larger cities are scheduled first so they don't bottleneck at the end.
    Returns {city: status}.
    """
    ordered = sorted(
        cities_and_attempts,
        key=lambda ca: CITY_CONFIGS.get(ca[0], DEFAULTS)['cpus'],
        reverse=True,
    )

    batch_results: dict = {}
    # max_workers >= len(ordered) so every city can block on the semaphore
    # concurrently; actual parallelism is capped by CPUPool
    with ThreadPoolExecutor(max_workers=len(ordered)) as executor:
        futures = {
            executor.submit(run_city, city, attempt): city
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
    timeout_label = f"{TIMEOUT_HOURS}h" if TIMEOUT_HOURS is not None else "none"
    _log(f"PlanHab runner: {len(CAPITAIS)} cities  CPU budget={MAX_CPU_BUDGET}  "
         f"timeout={timeout_label}  retries={MAX_RETRIES}")

    results = run_batch([(city, 1) for city in CAPITAIS])

    failed = [city for city, status in results.items() if status != "OK"]
    if failed and MAX_RETRIES > 0:
        sep = "─" * 55
        _log(f"\n{sep}\nRetrying {len(failed)} cities: {failed}\n{sep}")
        retry_results = run_batch([(city, 2) for city in failed])
        results.update(retry_results)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print("SUMMARY")
    print(f"{'═' * 55}")
    ok_count = 0
    for city in CAPITAIS:
        status = results.get(city, "NOT RUN")
        icon   = "v" if status == "OK" else "x"
        print(f"  [{icon}] {city:<22s}  {status}")
        if status == "OK":
            ok_count += 1
    print(f"{'─' * 55}")
    print(f"  {ok_count}/{len(CAPITAIS)} cities completed successfully.")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
