import os

for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import conf
import inspect
import tempfile
import numpy as np
import main
from simulation import Simulation

PASS = 0
FAIL = 0


def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS  {label}")
    else:
        FAIL += 1
        msg = f"  ({detail})" if detail else ""
        print(f"FAIL  {label}{msg}")


# ── shared short run ─────────────────────────────────────────────────────────
print("Initializing simulation (1 000-day run on ARACAJU @ 1%)...")
conf.RUN["TOTAL_DAYS"] = 1_000
conf.PARAMS["PROCESSING_ACPS"] = ["ARACAJU"]
conf.PARAMS["PERCENTAGE_ACTUAL_POP"] = 0.01

path = tempfile.gettempdir()
sim = Simulation(conf.PARAMS, path)
sim.initialize()

N_HOUSES_INIT = len(sim.houses)
sim.run()


# ── helpers ──────────────────────────────────────────────────────────────────
def gini_of_sim(s):
    incomes = np.array([f.get_permanent_income() for f in s.families.values()])
    incomes = incomes - incomes.min() + 1e-6
    n = len(incomes)
    if n == 0:
        return 0
    s_ = np.sort(incomes)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * s_) / (n * s_.sum()))


def vacancy_rate(s):
    houses = list(s.houses.values())
    if not houses:
        return 0
    return sum(1 for h in houses if h.family_id is None) / len(houses)


def unemployment_rate(s):
    return s.stats.global_unemployment_rate


# ── 1. STRUCTURAL INTEGRITY (original checks) ────────────────────────────────
print("\n── Structural integrity ─────────────────────────────────────────────")

check(
    "Construction increases housing supply",
    len(sim.houses) > N_HOUSES_INIT,
    f"init={N_HOUSES_INIT}, final={len(sim.houses)}",
)

check(
    "Bank is loaning money",
    sim.central.n_loans() > 0,
    f"loans={sim.central.n_loans()}",
)

check(
    "No families without a house",
    all(f.house is not None for f in sim.families.values()),
    f"homeless={sum(1 for f in sim.families.values() if f.house is None)}",
)

check(
    "No more than one family per house",
    len({f.house for f in sim.families.values()}) == len(sim.families),
)

# ── 2. ECONOMIC SANITY BOUNDS ────────────────────────────────────────────────
print("\n── Economic sanity bounds ───────────────────────────────────────────")

g = gini_of_sim(sim)
check(
    "Gini index in plausible range [0.30, 0.65]",
    0.30 <= g <= 0.65,
    f"gini={g:.4f}",
)

u = unemployment_rate(sim)
check(
    "Unemployment rate in plausible range [0.01, 0.35]",
    0.01 <= u <= 0.35,
    f"unemployment={u:.4f}",
)

v = vacancy_rate(sim)
check(
    "Housing vacancy rate in plausible range [0.02, 0.30]",
    0.02 <= v <= 0.30,
    f"vacancy={v:.4f}",
)

bank_balance = sim.central.balance
check(
    "Bank remains solvent (balance > 0)",
    bank_balance > 0,
    f"balance={bank_balance:.2f}",
)

zero_consumption = sum(
    1 for f in sim.families.values() if f.average_utility == 0
) / max(len(sim.families), 1)
check(
    "Zero-consumption families below 20%",
    zero_consumption < 0.20,
    f"zero_consumption_ratio={zero_consumption:.3f}",
)

# ── 3. MECHANISM-SPECIFIC REGRESSION TESTS ───────────────────────────────────
print("\n── Mechanism regression tests ───────────────────────────────────────")

# Government transfer fix: gov firms must have received revenue during the run.
gov_firms = [f for f in sim.firms.values() if f.sector == "Government"]
gov_with_revenue = sum(1 for f in gov_firms if f.revenue > 0)
check(
    "Government firms received revenue (transfer gate fixed)",
    gov_with_revenue > 0,
    f"gov_firms={len(gov_firms)}, with_revenue={gov_with_revenue}",
)

# Brasília cold-start fix: construction firms must have positive total_quantity balance.
construction_firms = [f for f in sim.firms.values() if f.sector == "Construction"]
construction_solvent = sum(1 for f in construction_firms if f.total_balance > 0)
check(
    "Construction firms financially active (cold-start fix)",
    construction_solvent > 0,
    f"construction_firms={len(construction_firms)}, solvent={construction_solvent}",
)

# Down-payment gate: buying families must have had savings ≥ 20% of house price.
# Proxy: any family that owns (not renting) should have a mortgage or prior savings;
# check that not every owner is a renter (i.e. some families bought houses).
owners = [f for f in sim.families.values() if not f.is_renting]
check(
    "Some families own their home (buy market is active)",
    len(owners) > 0,
    f"owners={len(owners)}",
)

# Rental market active: at least some families are renting.
renters = [f for f in sim.families.values() if f.is_renting]
check(
    "Rental market active (some families are renting)",
    len(renters) > 0,
    f"renters={len(renters)}",
)

# Wages being paid: agents should have non-zero last_wage on average.
employed = [a for a in sim.agents.values() if a.last_wage > 0]
check(
    "Labor market active (employed agents have positive wages)",
    len(employed) > 0,
    f"employed={len(employed)}/{len(sim.agents)}",
)

# ── sweep-safety guard ───────────────────────────────────────────────────────
# Sensitivity sweeps (main.py multiple_runs) override a per-run params dict that
# becomes sim.PARAMS; conf.PARAMS keeps its defaults. So any model code reading
# conf.PARAMS[...] silently ignores the swept value. This once voided an entire
# FUNDS_AVAILABILITY batch, which ran as N identical replications of the default.
# Read sim.PARAMS / self.params instead.
import pathlib
import re

_MODEL_DIRS = ["agents", "world", "markets", "analysis"]
_ALLOWED = {
    # Plotting reads it only as a fallback default, never as a swept value.
    "analysis/plotting/__init__.py",
}
_offenders = []
for _d in _MODEL_DIRS:
    for _f in pathlib.Path(_d).rglob("*.py"):
        _rel = _f.as_posix()
        if _rel in _ALLOWED:
            continue
        for _i, _line in enumerate(_f.read_text().splitlines(), 1):
            if re.search(r"\bconf\.PARAMS\s*\[", _line):
                _offenders.append(f"{_rel}:{_i}")

check(
    "No model code reads swept params from the conf.PARAMS module global",
    not _offenders,
    f"offenders={_offenders}",
)

# ── matched-seed guard ───────────────────────────────────────────────────────
# The exact-counterfactual design requires a treated run and its baseline to draw
# the same random numbers, so that the only difference between them is the policy.
# main.py hands one seed per replication to every configuration via PARAMS['SEED'];
# if the simulation stops honouring it, every difference silently picks up
# simulation noise and per-city effects lose power.
from simulation import resolve_seed  # noqa: E402

_p = dict(conf.PARAMS)
_p["SEED"] = 987654321
_resolved = [resolve_seed(dict(_p)) for _ in range(3)]
check(
    "PARAMS['SEED'] is honoured, so matched-seed differencing is exact",
    _resolved == [987654321] * 3,
    f"resolved={_resolved}",
)

_p.pop("SEED", None)
_free = [resolve_seed(dict(_p)) for _ in range(3)]
check(
    "Without an explicit seed, runs still vary under KEEP_RANDOM_SEED",
    len(set(_free)) == 3 if conf.RUN["KEEP_RANDOM_SEED"] else len(set(_free)) == 1,
    f"free={_free}",
)

# ── reproducibility guards ───────────────────────────────────────────────────
# A run must be reproducible from its seed. Three things break that: drawing from an
# unseeded global RNG, generating ids outside the seeded stream, and multithreaded
# BLAS reductions, whose summation order varies between processes.
_RNG_PAT = re.compile(r"\bnp\.random\.(?!RandomState)|\bnumpy\.random\.(?!RandomState)"
                      r"|(?<![.\w])random\.(random|randint|choice|sample|shuffle|uniform|gauss|normalvariate)"
                      r"|\buuid\.uuid[0-9]")
_rng_offenders = []
for _d in _MODEL_DIRS + ["."]:
    for _f in pathlib.Path(_d).glob("*.py") if _d == "." else pathlib.Path(_d).rglob("*.py"):
        _rel = _f.as_posix()
        if _rel.startswith("analysis/") and "planhab" not in _rel and "validation" not in _rel:
            pass
        if _rel.startswith("analysis/plotting") or "/emission_plots/" in _rel:
            continue
        if _rel.startswith("analysis/") and _rel not in ("analysis/stats.py", "analysis/output.py"):
            continue
        for _i, _line in enumerate(_f.read_text().splitlines(), 1):
            if _line.lstrip().startswith("#"):
                continue
            if _RNG_PAT.search(_line):
                _rng_offenders.append(f"{_rel}:{_i}")

check(
    "Model code draws only from the seeded RNG (no global np.random/random/uuid)",
    not _rng_offenders,
    f"offenders={_rng_offenders}",
)

check(
    "BLAS thread count pinned to 1 so float reductions are order-stable",
    all(os.environ.get(v) == "1" for v in
        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
    "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, "
    "or run through main.py which sets them",
)

# ── id generation vs. cached populations ─────────────────────────────────────
print("\n── Id generation ────────────────────────────────────────────────────")

_gen = sim.generator

check(
    "every id in the live population is unique across agents, houses, families, firms",
    len(set(sim.agents) | set(sim.houses) | set(sim.families) | set(sim.firms))
    == len(sim.agents) + len(sim.houses) + len(sim.families) + len(sim.firms),
    "an id shared by two objects makes lookup by id return the wrong one",
)

check(
    "house.owner_id resolves to a family that lists the house in owned_houses",
    all(h in sim.families[h.owner_id].owned_houses
        for h in sim.houses.values()
        if h.family_owner and h.owner_id in sim.families),
    "owner_id and owned_houses disagree, so owned_houses.remove(house) raises",
)

# A run that loads StoragedAgents gets a Generator whose counter starts at zero
# while the population already holds ids it would mint.
_cached = {'i%011d' % i for i in range(1, 51)}
_gen._next_id = 0
_gen.resume_ids(_cached)
_minted = {_gen.gen_id() for _ in range(20)}
check(
    "ids minted after loading a population do not collide with it",
    not (_minted & _cached),
    f"collisions={sorted(_minted & _cached)[:5]}",
)

_gen._next_id = 0
_gen.resume_ids({'0eaf0477-6be', 'f6611467-210'}, {})
check(
    "uuid-style and empty populations leave the counter alone",
    _gen._next_id == 0,
    f"_next_id={_gen._next_id}",
)

# ── QLI fiscal leg (defect #5) ───────────────────────────────────────────────
print("\n── QLI fiscal leg ───────────────────────────────────────────────────")

from collections import defaultdict  # noqa: E402
from agents.region import Region  # noqa: E402

_qli_params = dict(sim.PARAMS)
_qli_params.update({'QLI_GROWTH_RATE': 0.002, 'QLI_MAX': 1.0,
                    'QLI_GDP_NORM': 3.5, 'QLI_SPEND_NORM': 0.9})


def _delta(index, gdp_pc, spend_pc, w):
    """QLI increment for one month at weight w, off a bare Region."""
    r = Region.__new__(Region)
    r.index = index
    p = dict(_qli_params, QLI_TAX_WEIGHT=w)
    r.update_qli(gdp_pc, spend_pc, p)
    return r.index - index


# The whole point of QLI_TAX_WEIGHT is that the fiscal leg is a *designed arm*: at
# w = 0 the model must reproduce the GDP-only rule bit for bit whatever the spending
# is, so every pre-existing calibration and every batch baseline stays comparable.
_base = _delta(0.7, 4.0, 0.0, 0.0)
check(
    "at QLI_TAX_WEIGHT = 0 public spending cannot move QLI",
    _delta(0.7, 4.0, 99.0, 0.0) == _base and _delta(0.7, 4.0, 0.5, 0.0) == _base,
    "w=0 must be exactly the pre-defect-#5 GDP-only rule",
)

check(
    "the fiscal leg moves QLI once it is weighted in",
    _delta(0.7, 4.0, 2.0, 1.0) > _delta(0.7, 4.0, 0.2, 1.0),
    "two municipalities with equal GDP per capita must differ when spending differs; "
    "this is the place-based instrument Paper A otherwise lacks",
)

# Calibration identity: at spend_pc/gdp_pc == QLI_SPEND_NORM/QLI_GDP_NORM the two
# drivers coincide, so w does not shift the baseline. That is what makes the w = 1
# arm comparable with w = 0 rather than a different model.
_ratio = _qli_params['QLI_SPEND_NORM'] / _qli_params['QLI_GDP_NORM']
check(
    "at the calibrated spend/GDP ratio the two drivers coincide, so w is neutral",
    abs(_delta(0.7, 4.0, 4.0 * _ratio, 1.0) - _delta(0.7, 4.0, 0.0, 0.0)) < 1e-12,
    "QLI_SPEND_NORM = mean(spend_pc/gdp_pc) × QLI_GDP_NORM is what buys this",
)

# applied_treasure is a cumulative stock that is never reset, which is why it could
# not be used as the driver. The accumulator beside it must be a monthly FLOW.
_r = Region.__new__(Region)
_r.applied_treasure = defaultdict(int)
_r.update_applied_taxes(10.0, 'fpm')
_r.update_applied_taxes(5.0, 'equally')
_first = _r.take_applied_flow()
_r.update_applied_taxes(2.0, 'locally')
check(
    "public money applied is read as a monthly flow and cleared, not as a stock",
    _first == 15.0 and _r.take_applied_flow() == 2.0 and _r.take_applied_flow() == 0.0
    and _r.applied_treasure['fpm'] == 10.0,
    f"first={_first}, applied_treasure kept its cumulative meaning",
)

# Integration: the plumbing in Funds.invest_taxes must actually feed the driver. A
# leg fed by zeros would pass every unit test above and do nothing in a real run.
_spend_seen = [r.qli_spend_pc for r in sim.regions.values()]
_gdp_seen = [r.qli_gdp_pc for r in sim.regions.values()]
check(
    "the live run feeds both QLI drivers with non-zero per-capita flows",
    any(s > 0 for s in _spend_seen) and any(g > 0 for g in _gdp_seen),
    f"max spend_pc={max(_spend_seen, default=0):.4f}, "
    f"max gdp_pc={max(_gdp_seen, default=0):.4f}",
)

# Region instances are pickled into StoragedAgents, which carries no source hash, so
# a cache written before these attributes existed is unpickled without them.
check(
    "a Region unpickled from an older cache still has the new QLI attributes",
    all(hasattr(Region, a) for a in ('applied_flow', 'qli_gdp_pc', 'qli_spend_pc')),
    "class-level defaults are what keep pre-#5 .agents caches loadable",
)

check(
    "a failing job is abandoned rather than resubmitted for ever",
    isinstance(getattr(main, "MAX_JOB_ATTEMPTS", None), int)
    and main.MAX_JOB_ATTEMPTS >= 1
    and "MAX_JOB_ATTEMPTS" in inspect.getsource(main._run_jobs_parallel),
    "_run_jobs_parallel must cap per-job attempts, not just BrokenExecutor restarts",
)

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'─' * 50}")
print(f"Results: {PASS} PASS  |  {FAIL} FAIL  |  {PASS + FAIL} total")
if FAIL:
    raise SystemExit(1)
