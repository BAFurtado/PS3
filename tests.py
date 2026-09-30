import os

for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import conf
import inspect
import tempfile
import numpy as np
from collections import defaultdict
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
# TOTAL_DAYS is a model parameter; setting it in conf.RUN had no effect and the tests ran 30 years
conf.PARAMS["TOTAL_DAYS"] = 1_000
conf.PARAMS["PROCESSING_ACPS"] = ["ARACAJU"]
conf.PARAMS["PERCENTAGE_ACTUAL_POP"] = 0.01

# Own directory: parallel test runs (e.g. two worktrees) must not append to the same stats.csv
path = tempfile.mkdtemp(prefix="ps3_tests_")
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

# One job per worker. Unemployment is read from agent.firm_id, employment and wages from
# firm.employees; an agent listed by two firms is paid by both and produces for both.
_listings = [(aid, a, f) for f in sim.firms.values() for aid, a in f.employees.items()]
_listed_ids = [aid for aid, _, _ in _listings]
_mismatched = [aid for aid, a, f in _listings if a.firm_id != f.id]
check(
    "Every employee is on exactly one payroll, the firm its firm_id names",
    len(_listed_ids) == len(set(_listed_ids)) and not _mismatched
    and len(_listed_ids) == sum(1 for a in sim.agents.values() if a.firm_id is not None),
    f"listings={len(_listed_ids)}, distinct={len(set(_listed_ids))}, mismatched={len(_mismatched)}",
)

# The distance pass receives the candidates the qualification pass left. When that list
# is empty, it must hire no one rather than fall back to the full, just-hired pool.
_lm = sim.labor_market
_hired_before = [a for a in sim.agents.values() if a.firm_id is not None][:5]
_firm = next(f for f in sim.firms.values() if f.sector != "Government")
_saved_cands, _saved_emp = _lm.candidates, dict(_firm.employees)
_lm.candidates = list(_hired_before)
_lm.matching_firm_offers([(_firm, 1.0)], sim.PARAMS, cand_looking=[])
_rehired = [a.id for a in _hired_before if a.id in _firm.employees and a.id not in _saved_emp]
_lm.candidates = _saved_cands
check(
    "An empty still-looking list after the first matching pass hires no one",
    not _rehired,
    f"re-hired={_rehired}",
)

# A month with fewer than two posts returns before matching; it must still clear both lists, or next
# month's look_for_jobs appends to a stale pool (duplicates, agents who died in between).
_lm.available_postings = [_firm]
_lm.candidates = list(_hired_before[:1])
_lm.assign_post(0.05, None, sim.PARAMS)
check(
    "A month with fewer than two posts leaves no candidates or posts for the next month",
    _lm.candidates == [] and _lm.available_postings == [],
    f"candidates={len(_lm.candidates)}, posts={len(_lm.available_postings)}",
)

# Government headcount is exogenous (RAIS target, gov_hire_fire). Its profit is always negative, so the profit and
# insolvency rules of hire_fire used to fire its staff every month and empty it within a few years.
_gov = next(f for f in sim.firms.values() if f.sector == "Government" and f.employees)
_saved = (_gov.profit, _gov.increase_production, _gov.total_balance, dict(_gov.employees))
_gov.profit, _gov.increase_production, _gov.total_balance = -1.0, False, -1.0
_lm.hire_fire({_gov.id: _gov}, 1.0, gov_headcount_only=True)
_kept = len(_gov.employees) == len(_saved[3])
_gov.profit, _gov.increase_production, _gov.total_balance = _saved[:3]
for _a in _saved[3].values():
    _gov.add_employee(_a)
check(
    "hire_fire leaves a loss-making, insolvent Government firm's staff alone",
    _kept,
    f"employees {len(_saved[3])} -> {len(_gov.employees)}",
)

_gov_target = np.ceil(_lm.gov_employees[_lm.gov_employees.ano == sim.clock.year].qtde_vinc_ativos.sum()
                      * sim.PARAMS["PERCENTAGE_ACTUAL_POP"])
_gov_emp = sum(f.num_employees for f in sim.firms.values() if f.sector == "Government")
if sim.PARAMS.get("GOV_REVISED", False):
    # Balanced budget: every unit of public revenue ends as payroll transfer, purchase fund, investment fund, policy
    # money or input fund; the investment is also recorded as the regions' applied public money (bookkeeping, not a
    # second payment).
    _funds = sim.funds
    _gov_all = [f for f in sim.firms.values() if f.sector == "Government"]
    # These checks call settle_government_budget on made-up revenue; the state is restored at the end of the block
    _gov_attrs = ("total_balance", "revenue", "_transfer_current", "purchase_fund", "input_fund", "investment_fund",
                  "public_wage", "public_offer")
    _gov_state = {_f.id: {_a: getattr(_f, _a) for _a in _gov_attrs} for _f in _gov_all}
    _state_before = (_funds.external_public_funding, dict(_funds.gov_budget_diag))
    _snap = lambda: (sum(f._transfer_current for f in _gov_all), sum(f.purchase_fund for f in _gov_all),
                     sum(f.investment_fund for f in _gov_all), sum(_funds.policy_money.values()),
                     sum(r.applied_flow for r in sim.regions.values()), sum(f.input_fund for f in _gov_all))
    _before = _snap()
    _ext_before = _funds.external_public_funding
    for _i, _rid in enumerate(sim.regions):
        _funds.pending_public_money[_rid]["equally"] += 10.0 + _i
    # money in = revenue put in + what GOV_EXTERNAL_FUNDING paid from outside the ACP
    _put = sum(10.0 + _i for _i in range(len(sim.regions)))
    _funds.settle_government_budget(sim.regions)
    _put += _funds.external_public_funding - _ext_before
    _d = [a - b for a, b in zip(_snap(), _before)]
    check(
        "Balanced government budget neither creates nor loses money",
        abs(sum(_d[:4]) + _d[5] - _put) < 1e-6 * _put and _d[0] > 0 and _d[5] > 0
        and abs(_d[4] - _d[2]) < 1e-6 * _put,
        f"in={_put:.4f}, out={sum(_d[:4]) + _d[5]:.4f} (payroll {_d[0]:.2f}, purchases {_d[1]:.2f}, "
        f"inputs {_d[5]:.2f}, investment {_d[2]:.2f}, policy {_d[3]:.2f}, recorded for regions {_d[4]:.2f})",
    )
    # Public production inputs come from the input fund, never the start-up capital; unspent input money joins
    # the purchases, and what is spent counts as revenue (output at cost)
    _g = next(f for f in _gov_all if f.employees and f.inventory)
    _saved = (_g.total_balance, _g.input_fund, _g.purchase_fund, _g.revenue, dict(_g.input_inventory))
    _g.input_fund = 50.0
    _sector_map = defaultdict(list)
    for _f in sim.firms.values():
        _sector_map[_f.sector].append(_f)
    _g.buy_inputs(1e6, sim.regional_market, sim.firms, sim.seed, None, None, _sector_map)
    _res = (_g.total_balance - _saved[0], _g.input_fund, _g.input_cost, _g.purchase_fund - _saved[2],
            _g.revenue - _saved[3])
    _g.total_balance, _g.input_fund, _g.purchase_fund, _g.revenue = _saved[:4]
    _g.input_inventory.update(_saved[4])
    check(
        "Government buys its production inputs from the budget's input fund, not its capital",
        _res[0] == 0 and _res[1] == 0 and 0 < _res[2] and abs(_res[2] + _res[3] - 50.0) < 1e-6
        and abs(_res[4] - _res[2]) < 1e-9,
        f"capital change={_res[0]}, input cost={_res[2]:.3f}, to purchases={_res[3]:.3f}, revenue +{_res[4]:.3f}",
    )
    _gov_wage = [f.public_wage for f in _gov_all if f.employees]
    _priv = [f.wages_paid / f.num_employees for f in sim.firms.values()
             if f.sector != "Government" and f.num_employees > 0 and f.wages_paid > 0]
    check(
        "Public wage is set and positive for staffed Government firms",
        _gov_wage and min(_gov_wage) > 0,
        f"public wage range={min(_gov_wage, default=0):.3f}-{max(_gov_wage, default=0):.3f}, "
        f"private median={np.median(_priv) if _priv else 0:.3f}",
    )
    # GOV_WAGE_RULE 'premium': each municipality's target payroll is private pay per unit of qualification ** alpha
    # times one plus its level-weighted premium, for the qualification its Government firms employ; the offer seen by
    # job seekers is one plus the premium times the mean private wage, not the pay per worker
    _alpha = sim.PARAMS["PRODUCTIVITY_EXPONENT"]
    _bill, _heads, _quals = defaultdict(float), defaultdict(int), defaultdict(float)
    for _f in sim.firms.values():
        if _f.sector != "Government" and _f.num_employees > 0 and _f.wages_paid > 0:
            _bill[_f.region_id[:7]] += _f.wages_paid
            _heads[_f.region_id[:7]] += _f.num_employees
            _quals[_f.region_id[:7]] += _f.total_qualification(_alpha)
    _prem = {"federal": sim.PARAMS["GOV_PREMIUM_FEDERAL"], "estadual": sim.PARAMS["GOV_PREMIUM_STATE"],
             "municipal": sim.PARAMS["GOV_PREMIUM_MUNICIPAL"]}
    _dev, _markups = [], set()
    for _m, _v in _funds.gov_budget_diag.items():
        _gq = sum(_f.total_qualification(_alpha) for _f in _funds.mun_gov_firms[int(_m)])
        if not (_quals[_m] and _gq):
            continue
        _mk = 1 + sum(_funds.gov_levels[_m][_k] * _prem[_k] for _k in _prem)
        _markups.add(round(_mk, 3))
        _dev.append(abs(_v["target"] / (_bill[_m] / _quals[_m] * _gq) - _mk))
    _offers = [(_f.offer_wage(0.05, 1.0), _f.wage_base(0.05, 1.0)) for _f in _gov_all if _f.employees]
    if sim.PARAMS.get("GOV_WAGE_RULE") == "premium":
        check(
            "Public payroll is private pay per unit of qualification times the level-weighted premium",
            _dev and max(_dev) < 1e-9 and max(_markups) >= 1.0,
            f"municipalities={len(_dev)}, max deviation={max(_dev, default=0):.2e}, markups={sorted(_markups)}",
        )
        check(
            "Government ranks job posts on its offer, set apart from its pay per worker",
            _offers and all(_o > 0 for _o, _w in _offers) and any(abs(_o - _w) > 1e-9 for _o, _w in _offers),
            f"offer/pay pairs (first 3)={[(round(_o, 3), round(_w, 3)) for _o, _w in _offers[:3]]}",
        )
    # GOV_EXTERNAL_FUNDING: a municipality whose budget cannot pay its public payroll gets the shortfall from outside
    # the ACP only up to the non-municipal share of the cost; a purely municipal public sector gets nothing
    _mun = next(_m for _m, _v in _funds.gov_budget_diag.items() if _v["staff"] > 0)
    _saved_levels = _funds.gov_levels[_mun]
    _ext = []
    for _lv in ({"federal": 0.5, "estadual": 0.5, "municipal": 0.0}, {"federal": 0.0, "estadual": 0.0, "municipal": 1.0}):
        _funds.gov_levels[_mun] = _lv
        _e0 = _funds.external_public_funding
        _funds.pending_public_money[next(_r for _r in sim.regions if _r[:7] == _mun)]["equally"] += 1e-6
        _funds.settle_government_budget(sim.regions)
        _ext.append(_funds.external_public_funding - _e0)
    _funds.gov_levels[_mun] = _saved_levels
    for _f in _gov_all:
        for _a, _val in _gov_state[_f.id].items():
            setattr(_f, _a, _val)
    _funds.external_public_funding, _funds.gov_budget_diag = _state_before[0], _state_before[1]
    if sim.PARAMS.get("GOV_EXTERNAL_FUNDING", False):
        check(
            "An underfunded public payroll is paid from outside the ACP only for federal and state staff",
            _ext[0] > 0 and _ext[1] == 0,
            f"external funding: non-municipal {_ext[0]:.4f}, municipal-only {_ext[1]:.4f}",
        )

# Firm demography in stats.csv reconciles with the firm stock: entries - exits = change in the number of firms
from analysis.output import columns_for  # noqa: E402
import pandas as pd  # noqa: E402
_st = pd.read_csv(sim.output.stats_path, sep=";", header=None)
_st.columns = columns_for("stats", _st.shape[1])
_net = _st.firms_entered.iloc[1:].sum() - _st.firms_exited.iloc[1:].sum()
check(
    "Firm entries minus exits in stats.csv equal the change in the firm count",
    _net == _st.firms_count.iloc[-1] - _st.firms_count.iloc[0] and _st.firms_entered.sum() > 0
    and _st.firms_count.iloc[-1] == len(sim.firms),
    f"entered={_st.firms_entered.sum()}, exited={_st.firms_exited.sum()}, "
    f"count {_st.firms_count.iloc[0]} -> {_st.firms_count.iloc[-1]}, live={len(sim.firms)}",
)

# Money is created or destroyed only through the ledger channels (analysis/money.py). Every leak found by the
# 2026-09-29 money audit (#35-#42) showed up here as a drift of money_unexplained.
_unexplained = _st.money_unexplained.abs().max()
check(
    "The money stock changes only through the ledger channels (#35-#42)",
    _unexplained < 1e-6 * _st.money_total.max(),
    f"max |unexplained| = {_unexplained:.3g}, stock up to {_st.money_total.max():.3g}",
)

# Refused demand by buyer type (diagnostic) adds up to each firm's refused quantity, and the stats columns to
# firms_unmet_share
_by_ok = all(abs(sum(r[1] for r in f.demand_by_buyer.values()) - f.unmet_quantity) < 1e-9 * max(1, f.unmet_quantity)
             for f in sim.firms.values() if f.demand_by_buyer)
_b = ['household', 'government', 'input', 'external']
_d = _st[[f"demand_{b}" for b in _b]].sum(axis=1)
_u = _st[[f"unmet_{b}" for b in _b]].sum(axis=1)
check("Refused demand by buyer adds up to the firms' refused quantity and to firms_unmet_share",
      _by_ok and np.allclose(np.where(_d > 0, _u / _d.where(_d > 0, 1), 0), _st.firms_unmet_share)
      and _st.demand_household.iloc[-1] > 0 and _st.demand_input.iloc[-1] > 0,
      f"per firm {_by_ok}; last month shares by buyer "
      f"{[round(_st[f'unmet_{b}'].iloc[-1] / max(_st[f'demand_{b}'].iloc[-1], 1e-12), 3) for b in _b]}")

# Matching diagnostic: the household refusals the same sector's leftover stock could cover are between 0 and the
# refusals themselves, after the household round and at month end
_cov = _st[['unmet_household_coverable', 'unmet_household_coverable_end']]
check("Coverable household refusals are between 0 and unmet_household",
      bool((_cov.values >= 0).all() and (_cov.values <= _st[['unmet_household']].values * (1 + 1e-9) + 1e-9).all()),
      f"last month coverable / refused {round(_st.unmet_household_coverable.iloc[-1] / max(_st.unmet_household.iloc[-1], 1e-12), 3)}")

# FPM hands out exactly what was collected. It divided by the sum of the *distinct* municipal FPM values, so
# municipalities in the same FPM band counted once: ARACAJU at 1 % (6 municipalities) got 4-5 % more than was
# collected from 2011 (#41)
if sim.PARAMS.get("GOV_REVISED", False) and sim.PARAMS["FPM_DISTRIBUTION"]:
    _saved_pending = sim.funds.pending_public_money
    sim.funds.pending_public_money = defaultdict(lambda: defaultdict(float))
    _pop_mun = defaultdict(int)
    for _rid, _p in sim.reg_pops.items():
        _pop_mun[_rid[:7]] += _p
    sim.funds.distribute_fpm(100.0, sim.regions, sim.reg_pops, _pop_mun, 2012)
    _paid = sum(d['fpm'] for d in sim.funds.pending_public_money.values())
    sim.funds.pending_public_money = _saved_pending
    check("FPM distributes exactly the amount collected (#41)", abs(_paid - 100.0) < 1e-9, f"paid {_paid:.6f} of 100")

# Rent comes out of savings once, and a family short of money with no bank deposits still consumes what is left after
# rent (#35, #36)
_fam = next((f for f in sim.families.values() if f.is_renting and not f.rent_voucher and not f.have_loan
             and not sim.central.wallet.get(f) and f.members), None)
if _fam is not None:
    _saved = (_fam.savings, _fam.permanent_income, {k: m.money for k, m in _fam.members.items()})
    _rent = _fam.house.rent_data[0]
    _fam.savings, _fam.permanent_income = 0.0, 4 * _rent
    for _i, _m in enumerate(_fam.members.values()):
        _m.money = 2 * _rent if _i == 0 else 0.0
    _p = dict(sim.PARAMS, PUBLIC_TRANSIT_COST=0, PRIVATE_TRANSIT_COST=0, CONSUMPTION_PROPENSITY=1.0)
    _c = _fam.decision_on_consumption(sim.central, sim.clock.year, sim.clock.months, _p, sim.regions)
    _kept = _fam.savings
    _fam.savings, _fam.permanent_income = _saved[0], _saved[1]
    for _k, _m in _fam.members.items():
        _m.money = _saved[2][_k]
    check("Consumption leaves the rent in savings and spends the rest (#35, #36)",
          abs(_c - _rent) < 1e-9 and abs(_kept - _rent) < 1e-9, f"rent {_rent:.4f}, consumed {_c:.4f}, kept {_kept:.4f}")

# A firm that loses its last worker has no wage bill. The last month's value used to stay in wages_paid and keep
# lowering its profit and firm tax.
_emptied = next(f for f in sim.firms.values() if f.sector != "Government" and f.employees)
_saved_staff, _saved_wp = dict(_emptied.employees), _emptied.wages_paid
_emptied.wages_paid = 123.0
_emptied.employees.clear()
_emptied.make_payment(sim.regions, 0.05, sim.PARAMS["PRODUCTIVITY_EXPONENT"], sim.PARAMS["TAX_LABOR"],
                      sim.PARAMS["RELEVANCE_UNEMPLOYMENT_SALARIES"])
_stale = _emptied.wages_paid
_emptied.employees.update(_saved_staff)
_emptied.wages_paid = _saved_wp
check(
    "A firm with no employees records no wage bill",
    _stale == 0,
    f"wages_paid={_stale}",
)

# Not exact: in a tight labour market Government refills a little slower than it loses staff. Without the fix
# it tends to zero.
check(
    "Government headcount does not collapse below its RAIS target",
    _gov_emp >= 0.75 * _gov_target,
    f"government employees={_gov_emp}, target={_gov_target:.0f}",
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

# ── Trade with the rest of Brazil (#26, #27, #28) ────────────────────────────────────────────────────────────────
print("\n── Trade with the rest of Brazil ────────────────────────────────────")
import glob  # noqa: E402
import random  # noqa: E402
import pandas as pd  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from markets.goods import read_technical_matrix, RegionalMarket, External  # noqa: E402

# Local + imported inputs of every buying sector sum to the national coefficient, in every ACP. A transposed file
# (#26, Brasília) has these sums in its rows; NaN coefficients (#28, 8 small ACPs) break them.
_national = pd.read_csv('input/technical_matrix.csv').set_index('sector')
_bad = []
for _p in glob.glob('input/technical_matrices/*_matrix_io.json'):
    _acp = os.path.basename(_p)[:-len('_matrix_io.json')]
    _ll, _el, _le, _ee = read_technical_matrix(_acp)
    _sums = (_ll + _el).sum(axis=0) - _national.loc[_ll.index, _ll.columns].sum(axis=0)
    if _ll.isna().values.any() or _el.isna().values.any() or _sums.abs().max() > 1e-6:
        _bad.append(_acp)
check("every ACP matrix: local + imported inputs = national coefficients, no NaN", not _bad, f"{_bad[:5]}")

# IO_IMPORTS picks the block firms and Government import from: external->local when on, the old local->external
# block (~0) when off
_markets = {flag: RegionalMarket(SimpleNamespace(PARAMS=dict(sim.PARAMS, IO_IMPORTS=flag), geo=sim.geo))
            for flag in (False, True)}
_ll, _el, _le, _ee = read_technical_matrix(sim.geo.processing_acps)
check("IO_IMPORTS on: firms import from the external->local block",
      _markets[True].ext_local_matrix.equals(_el) and _el.values.sum() > _le.values.sum())
check("IO_IMPORTS off: firms read the local->external block, as the old model did",
      _markets[False].ext_local_matrix.equals(_le))


# EXTERNAL_RECYCLING_SHARE = 1 returns the whole net import bill as demand for local firms, and a share in (0, 1)
# leaves the rest as a deficit in net_position. Stub firms with ample stock, so the only limit is the money.
class _StubFirm:
    def __init__(self, sector):
        self.sector, self.region_id, self.total_quantity, self.prices, self.sold = sector, 'r', 1e9, 1.0, 0.0

    def sale(self, amount, *args, **kwargs):
        self.sold += amount
        return 0.0


def _external_after(share, imports=100.0):
    _firms = {i: _StubFirm(s) for i, s in enumerate(['Agriculture', 'Manufacturing'])}
    _market = SimpleNamespace(technical_matrix=pd.DataFrame(index=['Agriculture', 'Manufacturing']),
                              external_demand_multiplier={'Agriculture': 0.5, 'Manufacturing': 0.1})
    _stub = SimpleNamespace(PARAMS=dict(sim.PARAMS, EXTERNAL_RECYCLING_SHARE=share), firms=_firms,
                            regional_market=_market, regions={}, ledger=defaultdict(float))
    _ext = External(_stub, sim.PARAMS["TAXES_STRUCTURE"]["consumption_equal"])
    _ext.intermediate_consumption(imports)
    _net_imports = imports - _ext.import_tax_month
    _ext.final_consumption({'Agriculture': 10.0, 'Manufacturing': 10.0}, random.Random(0))
    return _ext, _net_imports, sum(f.sold for f in _firms.values())


_ext, _net, _sold = _external_after(1.0)
check("recycling share 1: the net import bill returns as demand, trade balanced",
      abs(_ext.last_month['recycled'] - _net) < 1e-9 and abs(_ext.net_position - _ext.last_month['exports']) < 1e-9
      and abs(_sold - _ext.last_month['exports'] - _net) < 1e-9,
      f"recycled={_ext.last_month['recycled']:.4f} net imports={_net:.4f} net_position={_ext.net_position:.4f}")
_ext, _net, _sold = _external_after(0.5)
check("recycling share 0.5: half the net import bill is a deficit in net_position",
      abs(_ext.net_position - (_ext.last_month['exports'] - 0.5 * _net)) < 1e-9)
_ext, _net, _sold = _external_after(0.0)
check("recycling share 0: exports only, the old model",
      _ext.last_month['recycled'] == 0 and abs(_sold - 0.5 * 10 - 0.1 * 10) < 1e-9)


# EXTERNAL_DEMAND_SPREAD = 'stock': the sector's external demand is split over every stocked firm by stock value, so
# nothing is refused while demand fits in that value. Stub firms sell at most their stock
class _StockedFirm(_StubFirm):
    def __init__(self, sector, quantity, price):
        super().__init__(sector)
        self.total_quantity, self.prices = quantity, price

    def sale(self, amount, *args, **kwargs):
        sold = min(amount, self.total_quantity * self.prices)
        self.sold += sold
        return amount - sold


_firms = {0: _StockedFirm('Agriculture', 1.0, 1.0), 1: _StockedFirm('Agriculture', 3.0, 2.0),
          2: _StockedFirm('Agriculture', 0.0, 1.0)}
_market = SimpleNamespace(technical_matrix=pd.DataFrame(index=['Agriculture']),
                          external_demand_multiplier={'Agriculture': 0.5})
_stub = SimpleNamespace(PARAMS=dict(sim.PARAMS, EXTERNAL_DEMAND_SPREAD='stock', EXTERNAL_RECYCLING_SHARE=0.0),
                        firms=_firms, regional_market=_market, regions={}, ledger=defaultdict(float))
_ext = External(_stub, sim.PARAMS["TAXES_STRUCTURE"]["consumption_equal"])
_ext.final_consumption({'Agriculture': 12.0}, random.Random(0))
check("EXTERNAL_DEMAND_SPREAD 'stock': demand split by stock value over every stocked firm, none refused",
      abs(_firms[0].sold - 6 / 7) < 1e-12 and abs(_firms[1].sold - 36 / 7) < 1e-12 and _firms[2].sold == 0
      and abs(_ext.last_month['exports'] - 6.0) < 1e-12,
      f"{[f.sold for f in _firms.values()]}, exports {_ext.last_month['exports']}")

# HOUSEHOLD_RETRY: a household short-served by the firm it picked tries the rest of its sample, in its strategy's order
# (price, or distance); off, the rest goes back to savings
from agents.family import Family  # noqa: E402


class _ShelfFirm:
    def __init__(self, fid, price, quantity):
        self.id, self.address = fid, None
        self.inventory = {0: SimpleNamespace(price=price, quantity=quantity)}

    def sale(self, amount, *args, **kwargs):
        p = self.inventory[0]
        q = min(amount / p.price, p.quantity)
        p.quantity -= q
        return amount - q * p.price


def _retry_case(retry, by_price):
    firms = [_ShelfFirm('a', 1.0, 1.0), _ShelfFirm('b', 2.0, 10.0), _ShelfFirm('c', 3.0, 0.0), _ShelfFirm('d', 4.0, 10.0)]
    # Distance order a, c, d, b: by distance the retry skips c (no stock) and buys from d
    house = SimpleNamespace(address=None, _firm_distances={'a': 1.0, 'c': 2.0, 'd': 3.0, 'b': 4.0})
    fam = SimpleNamespace(savings=0.0, house=house, region_id=None, average_utility=0.0,
                          decision_on_consumption=lambda *a: 5.0)
    rm = SimpleNamespace(final_demand={'HouseholdConsumption': {'Agriculture': 1.0}}, household_no_stock=0.0,
                         household_unserved=0.0, household_import_share={})
    seed = SimpleNamespace(randint=lambda a, b: int(by_price), sample=None)
    Family.consume(fam, rm, seed, None, None, {}, dict(sim.PARAMS, SIZE_MARKET=5, HOUSEHOLD_RETRY=retry), 2010, 1,
                   False, {'Agriculture': firms})
    return [round(10 - f.inventory[0].quantity, 9) if f.id != 'a' else round(1 - f.inventory[0].quantity, 9)
            for f in firms if f.id != 'c'], fam.savings, rm.household_unserved


_cases = {(r, p): _retry_case(r, p) for r in (False, True) for p in (True, False)}
check("HOUSEHOLD_RETRY: short-served households buy the rest from the next stocked firm of their sample; off unchanged",
      _cases[(False, True)] == ([1.0, 0.0, 0.0], 4.0, 4.0) and _cases[(False, False)] == ([1.0, 0.0, 0.0], 4.0, 4.0)
      and _cases[(True, True)] == ([1.0, 2.0, 0.0], 0.0, 0.0) and _cases[(True, False)] == ([1.0, 0.0, 1.0], 0.0, 0.0),
      f"{_cases}")

# HOUSEHOLD_IMPORTS: the import share of a product goes to the rest of Brazil at once, the rest to the local firm, and
# all of it counts as consumption; the market's share comes from the import block of the technical matrix
_imp_ext = External(SimpleNamespace(PARAMS=sim.PARAMS, ledger=defaultdict(float)), 0.0)
_imp_firm = _ShelfFirm('a', 1.0, 100.0)
_imp_fam = SimpleNamespace(savings=0.0, house=SimpleNamespace(address=None, _firm_distances={'a': 1.0}), region_id=None,
                           average_utility=0.0, decision_on_consumption=lambda *a: 10.0)
_imp_rm = SimpleNamespace(final_demand={'HouseholdConsumption': {'Agriculture': 0.6, 'Trade': 0.4}},
                          household_no_stock=0.0, household_unserved=0.0, household_imports=0.0,
                          household_import_share={'Agriculture': 0.25}, sim=SimpleNamespace(external=_imp_ext))
_imp_cons = Family.consume(_imp_fam, _imp_rm, SimpleNamespace(randint=lambda a, b: 1), None, None, {},
                           dict(sim.PARAMS, SIZE_MARKET=5), 2010, 1, False,
                           {'Agriculture': [_imp_firm], 'Trade': [_ShelfFirm('t', 1.0, 100.0)]})
_rm_on = RegionalMarket(SimpleNamespace(PARAMS=dict(sim.PARAMS, HOUSEHOLD_IMPORTS=True), geo=sim.geo))
_ll, _el = read_technical_matrix(sim.geo.processing_acps)[:2]
_m = (_el.sum(axis=1) / (_ll.sum(axis=1) + _el.sum(axis=1)))
check("HOUSEHOLD_IMPORTS: the tradable import share is bought outside and counted as consumption; services stay local",
      abs(_imp_ext.imports_month - 1.5) < 1e-12 and abs(_imp_rm.household_imports - 1.5) < 1e-12
      and abs(100 - _imp_firm.inventory[0].quantity - 4.5) < 1e-12 and abs(_imp_cons['Agriculture'] - 6.0) < 1e-12
      and abs(_imp_ext.sim.ledger['imports'] + 1.5) < 1e-12
      and set(_rm_on.household_import_share) <= set(sim.PARAMS['HOUSEHOLD_IMPORT_SECTORS'])
      and all(abs(_rm_on.household_import_share[k] - _m[k]) < 1e-12 for k in _rm_on.household_import_share)
      and sim.regional_market.household_import_share == {},
      f"imports {_imp_ext.imports_month}, local sold {100 - _imp_firm.inventory[0].quantity}, "
      f"shares {_rm_on.household_import_share}")

# ── Firm capital and demography (#21, #24). Last: these remove firms from the shared run ─────────────────────────
print("\n── Firm capital, entry and exit ─────────────────────────────────────")
from world.firms import fund_entrant, firm_exit  # noqa: E402

if sim.PARAMS["FIRM_CAPITAL_MONTHS"] > 0:
    _priv = [f for f in sim.firms.values() if f.sector != "Government"]
    _months = sum(f.total_balance for f in _priv) / max(sum(f.revenue for f in _priv), 1e-9)
    check(
        "Firm capital is months, not millennia, of revenue",
        _months < 120,
        f"private firm balances = {_months:.0f} months of this month's revenue (original sizing ~5,000)",
    )
    _gov_bal = sum(f.total_balance for f in sim.firms.values() if f.sector == "Government")
    _gov_pay = sum(f.wages_paid for f in sim.firms.values() if f.sector == "Government")
    check(
        "Government holds no start-up capital, only its budget in transit",
        _gov_bal <= 2 * _gov_pay + 1e-6,
        f"Government balances {_gov_bal:.2f} vs monthly payroll {_gov_pay:.2f}",
    )
    # Entry moves capital from the sector's incumbents to the entrant; it creates none
    _bal = lambda: sum(f.total_balance for f in sim.firms.values())
    _before, _n, _unf = _bal(), len(sim.firms), sim.firm_entry_unfunded
    _entered = [fund_entrant(sim, _r) for _r in list(sim.regions.values())[:20]]
    check(
        "Firm entry is funded by incumbents and creates no money",
        abs(_bal() - _before) < 1e-6 * max(_before, 1)
        and len(sim.firms) - _n + sim.firm_entry_unfunded - _unf == 20
        and all(e is None or e.sector != "Government" for e in _entered),
        f"balances {_before:.4f} -> {_bal():.4f}, entered {len(sim.firms) - _n}, "
        f"unfunded {sim.firm_entry_unfunded - _unf}",
    )

    # A builder recovers the land it bought from revenue before wages, so it can buy the next plot
    _b = next(f for f in sim.firms.values() if f.sector == "Construction" and f.employees)
    _saved_sched, _saved_rev = _b.land_schedule, _b.revenue
    _b.revenue = 1e3
    _b.land_schedule = None
    _w0 = _b.wage_base(0.05, sim.PARAMS["RELEVANCE_UNEMPLOYMENT_SALARIES"])
    _b.land_schedule = defaultdict(float, {_b.present: 12.0})
    _w1 = _b.wage_base(0.05, sim.PARAMS["RELEVANCE_UNEMPLOYMENT_SALARIES"])
    _ic = _b.input_cost
    _b.land_schedule, _b.revenue = _saved_sched, _saved_rev
    _share = np.exp(-0.05 * sim.PARAMS["RELEVANCE_UNEMPLOYMENT_SALARIES"])
    check(
        "A builder deducts this month's land recovery from its wage base, and not from input_cost",
        abs((_w0 - _w1) * _b.num_employees - 12.0 * _share) < 1e-9 and _ic == _b.input_cost,
        f"wage bill {_w0 * _b.num_employees:.3f} -> {_w1 * _b.num_employees:.3f}",
    )

if sim.PARAMS["FIRM_EXIT_MONTHS"] > 0:
    _exit_months = sim.PARAMS["FIRM_EXIT_MONTHS"]
    _cands = [f for f in sim.firms.values() if f.sector not in ("Government", "Construction") and f.employees]
    _broke = _cands[0]
    _idle = next(f for f in _cands[1:] if f.sector != _broke.sector
                 and sum(g.sector == f.sector for g in sim.firms.values()) > 1)
    _staff = list(_broke.employees.values())
    # Make them look like exits, and every structure that keeps firms hold them
    _broke.total_balance, _broke.months_insolvent = -2.0, _exit_months - 1
    _idle_heirs = [f for f in sim.firms.values() if f.sector == _idle.sector and f is not _idle]
    _heirs_before = sum(f.total_balance for f in _idle_heirs)
    for _a in list(_idle.employees.values()):
        _idle.employees.pop(_a.id)
        _a.firm_id = None
    _idle.amount_sold, _idle.months_idle, _idle.total_balance = 0, _exit_months - 1, 7.0
    _house = next(iter(sim.houses.values()))
    _house.distance_to_firm(_broke)
    sim.labor_market.available_postings = [_broke, _idle]
    _writeoff = sim.firm_exit_writeoff
    firm_exit(sim)
    _gone = [_broke, _idle]
    check(
        "An insolvent or idle firm exits into sim.firm_grave, with its date and reason",
        all(f.id not in sim.firms and sim.firm_grave.get(f.id) is f for f in _gone)
        and _broke.exit_reason == "insolvent" and _idle.exit_reason == "idle" and _broke.exit_date == sim.clock.days,
        f"reasons {_broke.exit_reason}/{_idle.exit_reason}",
    )
    check(
        "An exiting firm is removed from every structure that holds firms",
        all(a.firm_id is None for a in _staff) and not _broke.employees
        and not any(f.id in h._firm_distances for h in sim.houses.values() for f in _gone)
        and not any(f in _gone for f in sim.labor_market.available_postings)
        and not any(f in _gone for fs in sim.funds.mun_gov_firms.values() for f in fs),
        f"staff still attached: {sum(a.firm_id is not None for a in _staff)}",
    )
    check(
        "Exit conserves money: capital goes to the sector's firms, a negative balance is written off",
        abs(sum(f.total_balance for f in _idle_heirs) - _heirs_before - 7.0) < 1e-9
        and abs(sim.firm_exit_writeoff - _writeoff - 2.0) < 1e-9 and _idle.total_balance == 0,
        f"heirs +{sum(f.total_balance for f in _idle_heirs) - _heirs_before:.4f}, "
        f"written off {sim.firm_exit_writeoff - _writeoff:.4f}",
    )

# ── Money creation and unit-free statistics (#31-#33) ────────────────────────
print("\n── Money creation and unit-free statistics ──────────────────────────")
from types import SimpleNamespace  # noqa: E402
from markets.rentmarket import collect_rent  # noqa: E402
from world.demographics import birth  # noqa: E402

_baby = birth(sim)
sim.total_pop -= 1
check("A newborn holds no money (#31)", _baby.money == 0, f"money {_baby.money}")


class _RentFamily:
    def __init__(self, savings, deposit):
        self.savings, self.deposit, self.rent_voucher, self.received = savings, deposit, 0, 0.0

    def grab_savings(self, bank, y, m):
        s, self.savings, self.deposit = self.savings + self.deposit, 0, 0
        return s

    def update_balance(self, amount):
        self.received += amount


def _pay_rent(savings, deposit, rent=0.4):
    tenant, landlord, taxes = _RentFamily(savings, deposit), _RentFamily(0, 0), []
    house = SimpleNamespace(rent_data=[np.float64(rent)], family_id='t', owner_id='l', region_id='r')
    stub = SimpleNamespace(families={'t': tenant, 'l': landlord}, PARAMS={'TAX_LABOR': sim.PARAMS['TAX_LABOR']},
                           central=SimpleNamespace(wallet={tenant: True}), clock=sim.clock,
                           regions={'r': SimpleNamespace(collect_taxes=lambda a, k: taxes.append(a))})
    collect_rent([house], stub)
    return tenant.savings, landlord.received + sum(taxes), savings + deposit


_rent_ok = []
for _cash in (0.1, 0.3):
    _kept, _paid, _before = _pay_rent(_cash, 5.0)
    _rent_ok.append(abs(_paid - 0.4) < 1e-9 and abs(_kept + _paid - _before) < 1e-9)
check("Rent paid from bank deposits is paid in full and conserves money (#32)", all(_rent_ok), f"{_rent_ok}")

_renters = [f for f in sim.families.values() if f.is_renting and f.get_permanent_income() > 0]
if len(_renters) > 20:
    _pi, _renters[0].permanent_income = _renters[0].permanent_income, 0
    _with = sim.stats.calculate_families_metrics(_renters)["rent_burden_decis"]
    _renters[0].permanent_income = _pi
    _without = sim.stats.calculate_families_metrics(_renters[1:])["rent_burden_decis"]
    check("Zero-income renters are left out of the rent-burden deciles (#33)", np.allclose(_with, _without),
          f"{_with} vs {_without}")

_f = next(f for f in sim.firms.values() if f.sector not in ("Government", "Construction") and f.employees)
_prod = _f.inventory[0]
_saved = (_prod.quantity, _prod.price, _f.amount_sold, _f.unmet_quantity, _f.total_balance, _f.revenue, _f.prices,
          _f.increase_production, _f.workers_needed)
_prod.quantity, _prod.price, _f.unmet_quantity = 2.0, 1.0, 0.0
_change = _f.sale(5.0, sim.regions, 0.0, _f.region_id, True) + _f.sale(4.0, sim.regions, 0.0, _f.region_id, True)
check("Firm.sale records the quantity it refuses for lack of stock (DEMAND_SIGNAL_UNMET)",
      abs(_change - 7.0) < 1e-12 and abs(_f.unmet_quantity - 7.0) < 1e-12, f"change {_change}, unmet {_f.unmet_quantity}")
_cap = _f.total_qualification(sim.PARAMS["PRODUCTIVITY_EXPONENT"]) / sim.PARAMS["PRODUCTIVITY_MAGNITUDE_DIVISOR"]
_prod.quantity, _f.amount_sold, _f.unmet_quantity = 0.0, 0.0, 10 * _cap + 1
_signal = []
for _on in (False, True):
    _f.decision_on_prices_production(1, 0.1, np.random.RandomState(0), _prod.price,
                                     sim.PARAMS["PRODUCTIVITY_EXPONENT"], sim.PARAMS["PRODUCTIVITY_MAGNITUDE_DIVISOR"],
                                     inventory_target_ratio=0.2, demand_signal_unmet=_on)
    _signal.append((_f.increase_production, _f.workers_needed))
check("Refused demand asks for more workers only with DEMAND_SIGNAL_UNMET on",
      _signal[0][0] is False and _signal[1][0] is True and _signal[1][1] > 1, f"{_signal}")
# PRICE_DEMAND_RESPONSE θ: refused demand raises the price by θ × refused share, above the markup cap, and blocks the
# month's fall; θ = 0 keeps the old rule (an above-average, well-stocked firm lowers its price)
_theta = []
for _t in (0.0, 0.5):
    _prod.quantity, _prod.price, _f.amount_sold, _f.unmet_quantity = 1e12, 1.2, 1.0, 3.0
    _f.decision_on_prices_production(1, 0.1, np.random.RandomState(1), 1.0,
                                     sim.PARAMS["PRODUCTIVITY_EXPONENT"], sim.PARAMS["PRODUCTIVITY_MAGNITUDE_DIVISOR"],
                                     price_markup_cap=0.0875, price_demand_response=_t)
    _theta.append(_prod.price)
check("PRICE_DEMAND_RESPONSE: refused demand raises the price beyond the cap; off keeps the old fall",
      _theta[0] < 1.2 and abs(_theta[1] - 1.2 * (1 + 0.5 * 0.75)) < 1e-12, f"{_theta}")
(_prod.quantity, _prod.price, _f.amount_sold, _f.unmet_quantity, _f.total_balance, _f.revenue, _f.prices,
 _f.increase_production, _f.workers_needed) = _saved

# IMPORT_PRICE 'exogenous': imported inputs cost 1 + freight whatever local prices are, and buying them conserves
# money. An external coefficient of 0.1 per sector is set on one firm's column so that it imports.
from analysis.money import money_stock_total  # noqa: E402
_rm = sim.regional_market
_ext_col = _rm._ext_local_np[_f.sector].copy()
_rm._ext_local_np[_f.sector][:] = 0.1
_old_ip = sim.PARAMS.get('IMPORT_PRICE', 'local')
sim.PARAMS['IMPORT_PRICE'] = 'exogenous'
for _s in _f.input_inventory:
    _f.input_inventory[_s] = 0.0
_f.total_balance = 1e7
_inv0, _stock0, _ledger0 = dict(_f.input_inventory), money_stock_total(sim), sum(sim.ledger.values())
_imports0 = sim.external.imports_month
_sector_map = defaultdict(list)
for _g in sim.firms.values():
    _sector_map[_g.sector].append(_g)
_desired = 3.0
_f.buy_inputs(_desired, _rm, sim.firms, sim.seed, None, None, _sector_map)
_freight = 1 + sim.PARAMS['REGIONAL_FREIGHT_COST']
_d_stock = money_stock_total(sim) - _stock0
_d_ledger = sum(sim.ledger.values()) - _ledger0
_n = len(_rm._sector_order)
check("IMPORT_PRICE 'exogenous': inputs bought outside cost 1 + freight, and buying them conserves money",
      abs(_d_stock - _d_ledger) < 1e-6 and sim.external.imports_month - _imports0 >= _n * _desired * 0.1 * _freight - 1e-9
      and all(_f.input_inventory[_s] - _inv0[_s] >= _desired * 0.1 - 1e-9 for _s in _rm._sector_order),
      f"stock change {_d_stock:.6f} vs ledger {_d_ledger:.6f}, imports {sim.external.imports_month - _imports0:.4f}")
_rm._ext_local_np[_f.sector][:] = _ext_col
sim.PARAMS['IMPORT_PRICE'] = _old_ip

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'─' * 50}")
print(f"Results: {PASS} PASS  |  {FAIL} FAIL  |  {PASS + FAIL} total")
if FAIL:
    raise SystemExit(1)
