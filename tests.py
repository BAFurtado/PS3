import conf
import tempfile
import numpy as np
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
    "Unemployment rate in plausible range [0.02, 0.35]",
    0.02 <= u <= 0.35,
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

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"Results: {PASS} PASS  |  {FAIL} FAIL  |  {PASS + FAIL} total")
if FAIL:
    raise SystemExit(1)