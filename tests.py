import os

for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import conf
import inspect
import tempfile
import numpy as np
import pandas as pd
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

# A matching pass with no postings (PCT_DISTANCE_HIRING 0 or 1) hires no one and leaves every candidate looking
_lm.candidates = list(_hired_before[:3])
_still = _lm.matching_firm_offers([], sim.PARAMS, flag='qualification')
_none = _lm.matching_firm_offers([], sim.PARAMS, cand_looking=list(_hired_before[:3]))
_lm.candidates = []
check(
    "A matching pass with no postings leaves every candidate looking",
    _still == list(_hired_before[:3]) and _none is None,
    f"still={len(_still or [])}, second pass={_none}",
)

# Start-up cap: the first `keep` postings stay, at most max(1, needed - keep) of the others are drawn
_privates = [f for f in sim.firms.values() if f.sector != "Government"][:10]
_gov_post = next(f for f in sim.firms.values() if f.sector == "Government")
_lm.available_postings = [_gov_post] + _privates
_lm.cap_postings(1, 4)
_capped = list(_lm.available_postings)
_lm.available_postings = list(_privates)
_lm.cap_postings(0, 0)
_floor = len(_lm.available_postings)
_lm.available_postings = []
check(
    "Start-up posting cap keeps Government's posts, samples needed - keep private posts, at least one",
    _capped[0] is _gov_post and len(_capped) == 4 and set(_capped[1:]) <= set(_privates) and _floor == 1,
    f"capped={len(_capped)}, floor={_floor}",
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
    # Public investment in its base window (real_public_spending leaves it as the budget left it)
    _spend_saved = (_funds.gov_spending_months, _funds.gov_spending_base)
    _funds.gov_spending_months, _funds.gov_spending_base = defaultdict(list), {}
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
    # GOV_WAGE_RULE 'premium': each municipality's target payroll is its municipal staff's share at private pay per
    # unit of wage weight times one plus the municipal premium, for the wage weight its Government firms employ, plus
    # the cost of its federal and state staff; the offer seen by job seekers is not the pay per worker
    _alpha = sim.PARAMS["PRODUCTIVITY_EXPONENT"]
    _bill, _heads, _quals = defaultdict(float), defaultdict(int), defaultdict(float)
    for _f in sim.firms.values():
        if _f.sector != "Government" and not _f.own_account and _f.num_employees > 0 and _f.wages_paid > 0:
            _bill[_f.region_id[:7]] += _f.wages_paid
            _heads[_f.region_id[:7]] += _f.num_employees
            _quals[_f.region_id[:7]] += _f.total_wage_weight(_alpha)
    _dev = []
    for _m, _v in _funds.gov_budget_diag.items():
        _gq = sum(_f.total_wage_weight(_alpha) for _f in _funds.mun_gov_firms[int(_m)])
        if not (_quals[_m] and _gq):
            continue
        _w_mun = _funds.gov_levels[_m]["municipal"] * (1 + sim.PARAMS["GOV_PREMIUM_MUNICIPAL"])
        _expect = _w_mun * _bill[_m] / _quals[_m] * _gq + _v["outside"]
        _dev.append(abs(_v["target"] / _expect - 1))
    _offers = [(_f.offer_wage(0.05, 1.0), _f.wage_base(0.05, 1.0)) for _f in _gov_all if _f.employees]
    if sim.PARAMS.get("GOV_WAGE_RULE") == "premium":
        check(
            "Public payroll: municipal staff at private pay per unit of wage weight, plus federal and state staff",
            _dev and max(_dev) < 1e-9,
            f"municipalities={len(_dev)}, max relative deviation={max(_dev, default=0):.2e}",
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
    _saved_price = sim.avg_prices
    _saved_pay = {_f.id: _f.wages_paid for _f in sim.firms.values() if _f.sector != "Government"}
    # Once the reference private pay is fixed, the cost of federal and state staff is that reference times each
    # level's observed multiple times their share of the staff, whatever private pay and prices do afterwards
    _saved_gov_pay = (_funds.gov_pay, _funds.gov_pay_months, _funds.gov_pay_reference)
    _funds.gov_pay = {str(_r.cod_mun): {"federal": _r.federal, "estadual": _r.estadual}
                      for _r in pd.read_csv("input/gov_pay.csv", sep=";").itertuples()}
    _funds.gov_pay_months, _funds.gov_pay_reference = [], None
    _burn, _base = sim.PARAMS["GOV_PAY_BURN_IN"], sim.PARAMS["GOV_PAY_BASE_MONTHS"]
    _refs = [_funds.national_pay_reference(float(_i)) for _i in range(_burn + _base + 3)]
    _frozen = np.mean(range(_burn, _burn + _base))
    _nat = []
    for _scale in (1.0, 2.0):
        for _f in sim.firms.values():
            if _f.sector != "Government":
                _f.wages_paid = _saved_pay[_f.id] * _scale
        sim.avg_prices = _saved_price * _scale
        for _rid in sim.regions:
            _funds.pending_public_money[_rid]["equally"] += 1e-6
        _funds.settle_government_budget(sim.regions)
        _nat.append((sum(_v["outside"] or 0 for _v in _funds.gov_budget_diag.values()),
                     sum(_v["target"] for _v in _funds.gov_budget_diag.values())))
    _expected = _frozen * sum(_v["staff"] * sum(_funds.gov_levels[_m][_k] * _funds.gov_pay[_m][_k]
                                                for _k in ("federal", "estadual"))
                              for _m, _v in _funds.gov_budget_diag.items())
    _funds.gov_pay, _funds.gov_pay_months, _funds.gov_pay_reference = _saved_gov_pay
    for _f in sim.firms.values():
        if _f.sector != "Government":
            _f.wages_paid = _saved_pay[_f.id]
    sim.avg_prices = _saved_price
    check(
        "Federal and state staff at the observed multiple of a private pay reference fixed after the base months",
        _refs[:_burn + _base - 1] == [float(_i) for _i in range(_burn + _base - 1)]
        and all(_r == _frozen for _r in _refs[_burn + _base - 1:])
        and _nat[0][0] > 0 and abs(_nat[0][0] - _expected) < 1e-9 * _expected
        and abs(_nat[1][0] - _nat[0][0]) < 1e-9 * _nat[0][0] and _nat[1][1] > _nat[0][1],
        f"reference {_frozen:.1f}, outside cost {_nat[0][0]:.3f} -> {_nat[1][0]:.3f} (expected {_expected:.3f}), "
        f"payroll target "
        f"{_nat[0][1]:.3f} -> {_nat[1][1]:.3f}",
    )
    for _f in _gov_all:
        for _a, _val in _gov_state[_f.id].items():
            setattr(_f, _a, _val)
    _funds.external_public_funding, _funds.gov_budget_diag = _state_before[0], _state_before[1]
    _funds.gov_spending_months, _funds.gov_spending_base = _spend_saved
    if sim.PARAMS.get("GOV_EXTERNAL_FUNDING", False):
        check(
            "An underfunded public payroll is paid from outside the ACP only for federal and state staff",
            _ext[0] > 0 and _ext[1] == 0,
            f"external funding: non-municipal {_ext[0]:.4f}, municipal-only {_ext[1]:.4f}",
        )

# The price index averages the prices of the firms with staff, stocked out or not, own-account pools left out
_pi_saved = sim.stats.previous_month_price
_pi = sim.stats.update_price(sim.firms, mid_simulation_calculus=True)[0]
sim.stats.previous_month_price = _pi_saved
_pi_stf = [i.price for f in sim.firms.values() for i in f.inventory.values() if f.num_employees > 0 and not f.pool]
_gp = sim.stats.group_prices(sim.firms, {"Agriculture", "Mining", "Manufacturing"})
_gp_t = [i.price for f in sim.firms.values() for i in f.inventory.values()
         if f.num_employees > 0 and not f.pool and f.sector in ("Agriculture", "Mining", "Manufacturing")]
_gp_n = [i.price for f in sim.firms.values() for i in f.inventory.values()
         if f.num_employees > 0 and not f.pool and f.sector not in ("Agriculture", "Mining", "Manufacturing")]
check("Tradable and non-tradable average prices split the firms of the price index",
      abs(_gp[0] - (np.mean(_gp_t) if _gp_t else 0)) < 1e-12 and abs(_gp[1] - np.mean(_gp_n)) < 1e-12,
      f"tradable {_gp[0]:.4f} ({len(_gp_t)}), non-tradable {_gp[1]:.4f} ({len(_gp_n)})")
check(
    "Price index averages the staffed firms",
    abs(_pi - np.mean(_pi_stf)) < 1e-12,
    f"index {_pi:.4f} over {len(_pi_stf)} firms",
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

# Refused demand by buyer type (diagnostic) adds up to each firm's refused quantity; the stats columns cover
# households, government, inputs and exports (investment purchases are in firms_unmet_share only)
_by_ok = all(abs(sum(r[1] for r in f.demand_by_buyer.values()) - f.unmet_quantity) < 1e-9 * max(1, f.unmet_quantity)
             for f in sim.firms.values() if f.demand_by_buyer)
_b = ['household', 'government', 'input', 'external']
_d = _st[[f"demand_{b}" for b in _b]].sum(axis=1)
_u = _st[[f"unmet_{b}" for b in _b]].sum(axis=1)
check("Refused demand by buyer adds up to the firms' refused quantity, unmet within demand by buyer",
      _by_ok and all((_st[f"unmet_{b}"] <= _st[f"demand_{b}"] + 1e-9).all() for b in _b)
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

# Stub firms with ample stock, so the only limit is the money
class _StubFirm:
    def __init__(self, sector):
        self.sector, self.region_id, self.total_quantity, self.prices, self.sold = sector, 'r', 1e9, 1.0, 0.0

    def sale(self, amount, *args, **kwargs):
        self.sold += amount
        return 0.0


# Exports are split over every stocked firm of the sector by stock value, so nothing is refused while demand fits in
# that value. Stub firms sell at most their stock
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
# Own-account pools with no members: nothing is paid to them
_NO_POOLS = SimpleNamespace(payable=lambda sector: (None, 0.0))
_market = SimpleNamespace(technical_matrix=pd.DataFrame(index=['Agriculture']), pools=_NO_POOLS,
                          input_need=np.zeros(1))
_stub = SimpleNamespace(PARAMS=sim.PARAMS, firms=_firms, regional_market=_market, regions={}, ledger=defaultdict(float))
_ext = External(_stub, sim.PARAMS["TAXES_STRUCTURE"]["consumption_equal"])
_ext.export_demand = lambda: {'Agriculture': 6.0}
_ext.final_consumption()
check("Exports split by stock value over every stocked firm, none refused",
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
    rm = SimpleNamespace(final_demand={'HouseholdConsumption': {'Trade': 1.0}}, household_no_stock=0.0,
                         household_unserved=0.0, household_import_share={}, monthly_hh_intended=defaultdict(float),
                         pools=_NO_POOLS)
    seed = SimpleNamespace(randint=lambda a, b: int(by_price), sample=None)
    Family.consume(fam, rm, seed, None, None, {}, dict(sim.PARAMS, SIZE_MARKET=5, HOUSEHOLD_RETRY=retry), 2010, 1,
                   False, {'Trade': firms})
    return [round(10 - f.inventory[0].quantity, 9) if f.id != 'a' else round(1 - f.inventory[0].quantity, 9)
            for f in firms if f.id != 'c'], fam.savings, rm.household_unserved


_cases = {(r, p): _retry_case(r, p) for r in (False, True) for p in (True, False)}
check("HOUSEHOLD_RETRY: short-served households buy the rest from the next stocked firm of their sample; off unchanged",
      _cases[(False, True)] == ([1.0, 0.0, 0.0], 4.0, 4.0) and _cases[(False, False)] == ([1.0, 0.0, 0.0], 4.0, 4.0)
      and _cases[(True, True)] == ([1.0, 2.0, 0.0], 0.0, 0.0) and _cases[(True, False)] == ([1.0, 0.0, 1.0], 0.0, 0.0),
      f"{_cases}")

# Tradable spending no local firm served (refused, or no stocked firm) is bought outside at 1 plus freight and counted
# as consumption; non-tradable spending refused goes back to savings
_sh_ext = External(SimpleNamespace(PARAMS=sim.PARAMS, ledger=defaultdict(float)), 0.0)
_sh_fam = SimpleNamespace(savings=0.0, house=SimpleNamespace(address=None, _firm_distances={'a': 1.0, 't': 1.0}),
                          region_id=None, average_utility=0.0, decision_on_consumption=lambda *a: 10.0)
_sh_rm = SimpleNamespace(final_demand={'HouseholdConsumption': {'Agriculture': 0.4, 'Manufacturing': 0.2, 'Trade': 0.4}},
                         household_no_stock=0.0, household_unserved=0.0, household_imports=0.0,
                         household_import_share={}, sim=SimpleNamespace(external=_sh_ext),
                         monthly_hh_intended=defaultdict(float), pools=_NO_POOLS)
_sh_cons = Family.consume(_sh_fam, _sh_rm, SimpleNamespace(randint=lambda a, b: 1), None, None, {},
                          dict(sim.PARAMS, SIZE_MARKET=5), 2010, 1, False,
                          {'Agriculture': [_ShelfFirm('a', 1.0, 2.0)], 'Trade': [_ShelfFirm('t', 1.0, 1.0)]})
# Government: with no stocked Manufacturing firm the whole purchase is imported and nothing stays in the fund
_ext = sim.external
_ext_saved = {k: v for k, v in vars(_ext).items() if isinstance(v, (int, float))}
_led_saved = sim.ledger['imports']
_manu = [f for f in sim.firms.values() if f.sector == 'Manufacturing']
_manu_q = [f.total_quantity for f in _manu]
for _f in _manu:
    _f.total_quantity = 0.0
_gf = next(f for f in sim.firms.values() if f.sector == 'Government')
_sh_tc = defaultdict(float)
_sh_rm_gov = SimpleNamespace(government_import_share={}, monthly_gov_intended=defaultdict(float), pools=_NO_POOLS)
_sh_left = _gf.spend_fund(type("S", (), {"firms": sim.firms, "seed": sim.seed, "regions": sim.regions,
                                         "external": _ext, "regional_market": _sh_rm_gov,
                                         "PARAMS": sim.PARAMS})(),
                          5.0, pd.Series({'Manufacturing': 1.0}), _sh_tc)
_sh_gov_imports = _ext.imports_month - _ext_saved['imports_month']
for _f, _q in zip(_manu, _manu_q):
    _f.total_quantity = _q
for _k, _v in _ext_saved.items():
    setattr(_ext, _k, _v)
sim.ledger['imports'] = _led_saved
check("Unserved tradable spending is imported (households and government); services go to savings",
      abs(_sh_ext.imports_month - 4.0) < 1e-12 and abs(_sh_rm.household_imports - 4.0) < 1e-12
      and abs(_sh_fam.savings - 3.0) < 1e-12 and abs(_sh_rm.household_unserved - 3.0) < 1e-12
      and abs(_sh_cons['Agriculture'] - 4.0) < 1e-12 and abs(_sh_cons['Manufacturing'] - 2.0) < 1e-12
      and abs(_sh_cons['Trade'] - 1.0) < 1e-12
      and _sh_left == 0.0 and abs(_sh_gov_imports - 5.0) < 1e-12 and abs(_sh_tc['Manufacturing'] - 5.0) < 1e-12,
      f"household imports {_sh_ext.imports_month}, savings {_sh_fam.savings}, consumption {dict(_sh_cons)}; "
      f"government left {_sh_left}, imports {_sh_gov_imports}")

# Households: the import share of a product goes to the rest of Brazil at once, the rest to the local firm, and all of
# it counts as consumption
_imp_ext = External(SimpleNamespace(PARAMS=sim.PARAMS, ledger=defaultdict(float)), 0.0)
_imp_firm = _ShelfFirm('a', 1.0, 100.0)
_imp_fam = SimpleNamespace(savings=0.0, house=SimpleNamespace(address=None, _firm_distances={'a': 1.0}), region_id=None,
                           average_utility=0.0, decision_on_consumption=lambda *a: 10.0)
_imp_rm = SimpleNamespace(final_demand={'HouseholdConsumption': {'Agriculture': 0.6, 'Trade': 0.4}},
                          household_no_stock=0.0, household_unserved=0.0, household_imports=0.0,
                          household_import_share={'Agriculture': 0.25}, sim=SimpleNamespace(external=_imp_ext),
                          monthly_hh_intended=defaultdict(float), pools=_NO_POOLS)
_imp_cons = Family.consume(_imp_fam, _imp_rm, SimpleNamespace(randint=lambda a, b: 1), None, None, {},
                           dict(sim.PARAMS, SIZE_MARKET=5), 2010, 1, False,
                           {'Agriculture': [_imp_firm], 'Trade': [_ShelfFirm('t', 1.0, 100.0)]})
check("Households buy the import share outside and count it as consumption; the rest is bought locally",
      abs(_imp_ext.imports_month - 1.5) < 1e-12 and abs(_imp_rm.household_imports - 1.5) < 1e-12
      and abs(100 - _imp_firm.inventory[0].quantity - 4.5) < 1e-12 and abs(_imp_cons['Agriculture'] - 6.0) < 1e-12
      and abs(_imp_ext.sim.ledger['imports'] + 1.5) < 1e-12,
      f"imports {_imp_ext.imports_month}, local sold {100 - _imp_firm.inventory[0].quantity}")

# No household demand for Real Estate (rent is paid in the rental market), the other shares rescaled in proportion
_hh_file = pd.read_csv('input/final_demand.csv').set_index('sector')['HouseholdConsumption']
_hh_off = sim.regional_market.final_demand['HouseholdConsumption']
_rest = _hh_file.drop('RealEstate')
check("Household demand: Real Estate share 0, others rescaled to sum 1",
      _hh_off['RealEstate'] == 0 and abs(_hh_off.sum() - 1) < 1e-12
      and np.allclose(_hh_off[_rest.index], _rest / _rest.sum()),
      f"{_hh_off.round(4).to_dict()}")

# Interregional trade: local + imported coefficients are the national ones split by the local share; households
# and government import 1 - share. The month-1 base: share = potential x min(output / demand, 1), exports = output -
# share x demand (none for Construction, Government); exports then = quantity x national growth x price ** (1 - sigma)
_io_params = dict(sim.PARAMS)
_io_rm = RegionalMarket(SimpleNamespace(PARAMS=_io_params, geo=sim.geo))
_io_sim = SimpleNamespace(PARAMS=_io_params, regional_market=_io_rm, firms=sim.firms, clock=sim.clock,
                          ledger=defaultdict(float), investment_rate=0.0, families={})
_io_ext = External(_io_sim, 0.0)
_nat = pd.read_csv('input/technical_matrix.csv').set_index('sector').loc[_io_rm._sector_order, _io_rm._sector_order]
_F = pd.Series(_io_params['TRADE_POTENTIAL'])[_io_rm._sector_order]
_io_split_ok = (np.allclose(_io_rm.technical_matrix + _io_rm.ext_local_matrix, _nat)
                and np.allclose(_io_rm.technical_matrix, _nat.mul(_F, axis=0))
                and all(abs(_io_rm.household_import_share[k] - (1 - _F[k])) < 1e-12 for k in _F.index)
                and _io_rm.government_import_share == _io_rm.household_import_share)
_cap_saved = {f.id: getattr(f, 'last_capacity', 0.0) for f in sim.firms.values()}
_io_sectors = defaultdict(list)
for _f in sim.firms.values():
    _f.last_capacity = 2.0
    _io_sectors[_f.sector].append(_f)
_io_rm.input_need[:] = 1.0
for _s in _io_rm._sector_order:
    _io_rm.monthly_hh_intended[_s] = 3.0
_io_rm.monthly_gov_intended['Trade'] = 4.0
_io_rm.monthly_fares = 5.0
_io_tab = _io_ext.trade_base()
_io_exp = {}
for _s in _io_rm._sector_order:
    _fs = _io_sectors.get(_s, [])
    _p = External.sector_price(_fs) if _fs else 1.0
    _q = 2.0 * len(_fs)
    _d = 1.0 + (3.0 + (4.0 if _s == 'Trade' else 0.0) + (5.0 if _s == 'Transport' else 0.0)) / _p
    _sh = _F[_s] if _s in ('Construction', 'Government') else _F[_s] * min(_q / _d, 1.0)
    _io_exp[_s] = (_sh, 0.0 if _s in ('Construction', 'Government') else _q - _sh * _d)
_io_base_ok = all(abs(_io_tab.loc[_s, 'local_share'] - v[0]) < 1e-12 and abs(_io_tab.loc[_s, 'exports'] - v[1]) < 1e-9
                  for _s, v in _io_exp.items())
_io_shares_ok = all(abs(_io_rm.household_import_share.get(_s, 0.0) - (1 - v[0])) < 1e-12 for _s, v in _io_exp.items())
# The rebased trade base: the same base with household spending and fares halved
_io_half = _io_ext.apply_trade_base(0.5, 'trade_base_test.csv')
_io_half_ok = True
for _s in _io_rm._sector_order:
    _fs = _io_sectors.get(_s, [])
    _p = External.sector_price(_fs) if _fs else 1.0
    _q = 2.0 * len(_fs)
    _d = 1.0 + (1.5 + (4.0 if _s == 'Trade' else 0.0) + (2.5 if _s == 'Transport' else 0.0)) / _p
    _sh = _F[_s] if _s in ('Construction', 'Government') else _F[_s] * min(_q / _d, 1.0)
    _ex = 0.0 if _s in ('Construction', 'Government') else _q - _sh * _d
    _io_half_ok &= abs(_io_half.loc[_s, 'local_share'] - _sh) < 1e-12 and abs(_io_half.loc[_s, 'exports'] - _ex) < 1e-9
_io_ext.apply_trade_base(1.0, 'trade_base_test.csv')
_io_ext.months = 5
_io_sigma = 0.5
_io_ext.sim.PARAMS = dict(_io_params, EXPORTS_PRICE_ELASTICITY=_io_sigma)
_io_dem = _io_ext.export_demand()
_io_dem_ok = all(abs(_io_dem.get(_s, 0.0) - (_io_exp[_s][1] * External.sector_price(_io_sectors[_s]) ** (1 - _io_sigma)
                                                if _io_exp[_s][1] > 0 and _io_sectors.get(_s) else 0.0)) < 1e-9
                 for _s in _io_rm._sector_order)
for _f in sim.firms.values():
    _f.last_capacity = _cap_saved[_f.id]
check("Interregional trade: national coefficients split by the local share; month-1 base sets shares and "
      "exports; exports at base quantity x price ** (1 - sigma)",
      _io_split_ok and _io_base_ok and _io_half_ok and _io_shares_ok and _io_dem_ok,
      f"split {_io_split_ok}, base {_io_base_ok}, shares {_io_shares_ok}, exports {_io_dem_ok}\n"
      f"{_io_tab.round(3).to_string()}")
# Government buys the import share outside, the rest locally, and records what it meant to spend
_gov_rm = SimpleNamespace(government_import_share={'Trade': 0.25}, monthly_gov_intended=defaultdict(float),
                          pools=_NO_POOLS)
_gov_ext = External(SimpleNamespace(PARAMS=sim.PARAMS, ledger=defaultdict(float)), 0.0)
_gov_tc = defaultdict(float)
_gov_shop = _ShelfFirm('t', 1.0, 4.0)
_gov_shop.sector, _gov_shop.total_quantity, _gov_shop.prices = 'Trade', 4.0, 1.0
_gov_left = next(f for f in sim.firms.values() if f.sector == 'Government').spend_fund(
    type("S", (), {"firms": {'t': _gov_shop}, "seed": sim.seed, "regions": sim.regions,
                   "external": _gov_ext, "regional_market": _gov_rm, "PARAMS": sim.PARAMS})(),
    8.0, pd.Series({'Trade': 1.0}), _gov_tc)
check("Government imports its share of each purchase",
      abs(_gov_ext.imports_month - 2.0) < 1e-12 and abs(_gov_rm.monthly_gov_intended['Trade'] - 8.0) < 1e-12
      and abs(_gov_tc['Trade'] - 6.0) < 1e-12 and abs(_gov_left - 2.0) < 1e-12,
      f"imports {_gov_ext.imports_month}, consumption {dict(_gov_tc)}, left {_gov_left}")

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
    # An entrant's capital comes from owners outside the ACP, recorded in the ledger
    _bal = lambda: sum(f.total_balance for f in sim.firms.values())
    _before, _n, _unf, _led = _bal(), len(sim.firms), sim.firm_entry_unfunded, sim.ledger['firm_entry']
    _entered = [fund_entrant(sim, _r) for _r in list(sim.regions.values())[:20]]
    check(
        "Firm entry is funded from outside the ACP through the ledger",
        abs(_bal() - _before - (sim.ledger['firm_entry'] - _led)) < 1e-6 * max(_before, 1)
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
    _share = type(_b).wage_shares['Construction']
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
# Tradables: an absolute ceiling below avg × (1 + cap) stops a low-inventory firm's rise at the ceiling
_parity = []
for _ceil in (None, 1.3):
    _prod.quantity, _prod.price, _f.amount_sold, _f.unmet_quantity = 0.0, 1.3, 1e6, 0.0
    _f.decision_on_prices_production(1, 0.1, np.random.RandomState(2), 1.25,
                                     sim.PARAMS["PRODUCTIVITY_EXPONENT"], sim.PARAMS["PRODUCTIVITY_MAGNITUDE_DIVISOR"],
                                     price_markup_cap=0.0875, price_ceiling=_ceil)
    _parity.append(_prod.price)
check("Tradables: the import-parity ceiling caps the inventory-driven rise",
      1.3 < _parity[0] <= 1.25 * 1.0875 + 1e-12 and _parity[1] == 1.3, f"{_parity}")
(_prod.quantity, _prod.price, _f.amount_sold, _f.unmet_quantity, _f.total_balance, _f.revenue, _f.prices,
 _f.increase_production, _f.workers_needed) = _saved

# IMPORT_PRICE 'exogenous': imported inputs cost 1 whatever local prices are, and buying them conserves money. One
# firm's column is set to import 0.1 per sector and buy nothing locally.
from analysis.money import money_stock_total  # noqa: E402
from agents.firm import import_price, import_parity  # noqa: E402
_rm = sim.regional_market
_ext_col = _rm._ext_local_np[_f.sector].copy()
_loc_col = _rm._tech_np[_f.sector].copy()
_rm._ext_local_np[_f.sector][:] = 0.1
_rm._tech_np[_f.sector][:] = 0.0
_old_ip = sim.PARAMS.get('IMPORT_PRICE', 'local')
sim.PARAMS['IMPORT_PRICE'] = 'exogenous'
_sector_map = defaultdict(list)
for _g in sim.firms.values():
    _sector_map[_g.sector].append(_g)
_desired = 3.0
_n = len(_rm._sector_order)
_buy = {}
for _mode, _unit in (('margins', 1.0),):
    for _s in _f.input_inventory:
        _f.input_inventory[_s] = 0.0
    _f.total_balance = 1e7
    _inv0, _stock0, _ledger0 = dict(_f.input_inventory), money_stock_total(sim), sum(sim.ledger.values())
    _imports0 = sim.external.imports_month
    _f.buy_inputs(_desired, _rm, sim.firms, sim.seed, None, None, _sector_map)
    _d_stock = money_stock_total(sim) - _stock0
    _d_ledger = sum(sim.ledger.values()) - _ledger0
    _imp = sim.external.imports_month - _imports0
    _buy[_mode] = (abs(_d_stock - _d_ledger) < 1e-6 and abs(_imp - _n * _desired * 0.1 * _unit) < 1e-9
                   and all(abs(_f.input_inventory[_s] - _inv0[_s] - _desired * 0.1) < 1e-9 for _s in _rm._sector_order),
                   round(_imp, 6), round(_d_stock - _d_ledger, 9))
check("IMPORT_PRICE 'exogenous': inputs bought outside cost 1, and buying them conserves money", _buy['margins'][0],
      f"{_buy}")
_rm._ext_local_np[_f.sector][:] = _ext_col
_rm._tech_np[_f.sector][:] = _loc_col
sim.PARAMS['IMPORT_PRICE'] = _old_ip
# Import price and the import-parity ceiling per tradable sector
_margins = pd.read_csv('input/transport_margins.csv', sep=';').set_index('sector')['margin']
check("Imports at 1, import parity at 1 + the product's transport margin",
      import_price(sim.PARAMS) == 1.0
      and import_parity(sim.PARAMS) == {s: 1 + float(_margins[s]) for s in sim.PARAMS['TRADABLE_SECTORS']}
      and 0 < _margins['Manufacturing'] < 0.05,
      f"{import_parity(sim.PARAMS)}")
# Participation: agents 17-69 active with the Census share for their sex, age group and municipality, from a draw they
# keep; unemployment counts the active and those in a job; start-up hiring reads the Census share of the active
# without a job
from world.participation import Participation
_pt = Participation(sim.mun_to_regions, 7)
_pf = pd.read_csv('input/participation_2010.csv', sep=';')
_pf = _pf[_pf.cod_mun.isin([int(m) for m in sim.mun_to_regions]) & (_pf['pop'] > 0)]
_pa = [a for a in sim.agents.values() if 16 < a.age < 70]
_pact = [_pt.is_active(a) for a in _pa]
_pexp = np.mean([_pt.rates[(a.region_id[:7], a.gender, _pt.groups[np.searchsorted(_pt.groups, a.age, 'right') - 1])]
                 for a in _pa])
_pold = next(a for a in sim.agents.values() if a.age >= 70)
_pstable = [_pt.is_active(a) for a in _pa] == _pact and Participation(sim.mun_to_regions, 7).draw(_pa[0]) == _pt.draw(_pa[0])
_pt_saved = sim.participation
sim.stats.participation = _pt
_pu = sim.stats.update_unemployment(sim.agents.values())
_pinit = _pt.unemployment
sim.stats.participation = sim.participation = _pt_saved
_pforce = [a for a, act in zip(_pa, _pact) if act or a.firm_id is not None]
_pu_exp = sum(1 for a in _pforce if a.firm_id is None) / len(_pforce)
check("Participation: active share matches the Census rates, draws stable, 70+ inactive, unemployment over the labour "
      "force, start-up at the Census rate of the active",
      abs(np.mean(_pact) - _pexp) < 0.03 and _pstable and not _pt.is_active(_pold) and np.isclose(_pu, _pu_exp)
      and np.isclose(_pinit, 1 - _pf.employed.sum() / _pf.active.sum()) and 0.02 < _pinit < 0.2,
      f"active {np.mean(_pact):.3f} vs {_pexp:.3f}, u {_pu:.3f} vs {_pu_exp:.3f}, start {_pinit:.3f}")

# Programme funds: after the base months they follow the municipality's base real GDP times the national index, not
# its current GDP; FGTS and SBPE instalments leave the ACP, market ones stay in the bank, money conserved
_st = sim.stats
_st_saved = (_st.funds_months, _st.funds_base_sum, _st.funds_base, _st.national_gdp,
             _st.funds_burn_in, _st.funds_base_months, _st.last_gdp)
_st.funds_months, _st.funds_base_sum, _st.funds_base = 0, defaultdict(float), None
_st.national_gdp = pd.read_csv("input/national_real_gdp.csv", sep=";").set_index("year")["index"]
_st.funds_burn_in, _st.funds_base_months = 1, 2
_mun = next(iter(_st_saved[6]))
_st.last_gdp = defaultdict(float, {_mun: 10.0})
_before = []
for _ in range(3):
    _before.append(_st.funds_gdp(_mun, 2014))
    _st.update_funds_base(2011)
_st.last_gdp[_mun] = 99.0
_after = _st.funds_gdp(_mun, 2014)
_expect = 10.0 / _st.national_index(2011) * _st.national_index(2014)
(_st.funds_months, _st.funds_base_sum, _st.funds_base, _st.national_gdp, _st.funds_burn_in,
 _st.funds_base_months, _st.last_gdp) = _st_saved
check("Programme funds follow current GDP until the base is fixed, then base real GDP times the national index",
      all(v == 10.0 for v in _before) and abs(_after - _expect) < 1e-9, f"before {_before}, after {_after:.4f}")

from agents.bank import Loan  # noqa: E402
_bank = sim.central
_fam = next(f for f in sim.families.values() if f.house is not None)
_saved_loans, _saved_savings, _saved_have = _bank.loans, _fam.savings, _fam.have_loan
_paid = {}
for _flag in (True,):
    _bank.loans = defaultdict(list, {_fam.id: [Loan(12.0, 0.0, 12, _fam.house, loan_type="fgts", table_type="price"),
                                              Loan(12.0, 0.0, 12, _fam.house, loan_type="market", table_type="price")]})
    _fam.savings = 100.0
    _bal0, _stock0, _ledger0 = _bank.balance, money_stock_total(sim), sum(sim.ledger.values())
    _bank.collect_loan_payments(sim)
    _paid[_flag] = (_bank.balance - _bal0, money_stock_total(sim) - _stock0 - (sum(sim.ledger.values()) - _ledger0))
    _bank.balance = _bal0
    sim.ledger["fgts_sbpe_repaid"] = 0.0
_bank.loans, _fam.savings, _fam.have_loan = _saved_loans, _saved_savings, _saved_have
_bank.recompute_outstanding_market_loans()
check("FGTS instalments leave the ACP, market instalments stay in the bank, money conserved",
      abs(_paid[True][0] - 1.0) < 1e-9 and abs(_paid[True][1]) < 1e-9,
      f"bank balance change {_paid[True][0]:.4f}, unexplained {_paid[True][1]:.2e}")

# The bank: a deposit earns each month's rate net of tax, and settlement returns equity to its target with money
# conserved
import datetime as _dt  # noqa: E402
_bank = sim.central
_saved_bank = (_bank.wallet, _bank.balance, _bank.taxes, _bank.interest, _bank.equity_target, dict(sim.ledger))
_fam = next(iter(sim.families.values()))
_bank.wallet = defaultdict(list)
_stock0, _ledger0 = money_stock_total(sim), sum(sim.ledger.values())
_bank.deposit(_fam, 100.0, _dt.date(2011, 1, 1))
_fam.savings -= 100.0
_bank.equity_target = _bank.equity() - 5.0
_other = next(f for f in sim.families.values() if f is not _fam)
_bank.wallet[_other]
_rates = (0.01, -0.02)
for _r in _rates:
    _bank.interest = _r
    _bank.accrue_deposit_interest(_dt.date(2011, 2, 1))
_owed = _bank.sum_deposits(_fam)
_empty_kept = not _bank.wallet[_other]
_expect = 100.0 * (1 + 0.01 * (1 - _bank.tax_firm)) * (1 - 0.02)
_bank.settle_with_national_bank()
_eq_gap = _bank.equity() - _bank.equity_target
_paid = _bank.withdraw(_fam, 2011, 3)
_fam.savings += _paid
_cons = money_stock_total(sim) - _stock0 - (sum(sim.ledger.values()) - _ledger0)
_fam.savings -= _paid - 100.0
(_bank.wallet, _bank.balance, _bank.taxes, _bank.interest, _bank.equity_target), _ld = _saved_bank[:5], _saved_bank[5]
sim.ledger.clear()
sim.ledger.update(_ld)
check("Bank: deposits accrue monthly, withdrawal pays them, settlement restores equity, money conserved",
      _empty_kept and abs(_owed - _expect) < 1e-9 and abs(_paid - _expect) < 1e-9 and abs(_eq_gap) < 1e-9 and abs(_cons) < 1e-12 * max(1.0, _stock0),
      f"owed {_owed:.6f} vs {_expect:.6f}, paid {_paid:.6f}, equity gap {_eq_gap:.2e}, unexplained {_cons:.2e}")

# Wealth norm: a family above its liquid-wealth target also spends WEALTH_ADJUSTMENT of the excess, from its deposits
# if needed; below it, it cuts spending by the same share of the gap
_fam = next((f for f in sim.families.values() if (not f.is_renting or f.rent_voucher) and not f.have_loan
             and f.members), None)
if _fam is not None:
    _bank = sim.central
    _saved = (_fam.savings, _fam.permanent_income, {k: m.money for k, m in _fam.members.items()},
              list(_bank.wallet.get(_fam, [])), _bank.balance, _bank.taxes)
    _today = _dt.date(sim.clock.year, sim.clock.months, 1)

    def _norm_case(cash, deposits, burn_in=0):
        _fam.savings, _fam.permanent_income = 0.0, 10.0
        for _i, _m in enumerate(_fam.members.values()):
            _m.money = cash if _i == 0 else 0.0
        _bank.wallet.pop(_fam, None)
        if deposits:
            _bank.deposit(_fam, deposits, _today)
        _p = dict(sim.PARAMS, PUBLIC_TRANSIT_COST=0, PRIVATE_TRANSIT_COST=0, CONSUMPTION_PROPENSITY=1.0,
                  WEALTH_TARGET_MONTHS=6, WEALTH_ADJUSTMENT=1 / 24,
                  WEALTH_NORM_BURN_IN=burn_in)
        _c = _fam.decision_on_consumption(_bank, sim.clock.year, sim.clock.months, _p, sim.regions)
        _left = _fam.savings + _bank.sum_deposits(_fam)
        _bank.wallet.pop(_fam, None)
        return _c, _left

    _above = _norm_case(5.0, 200.0)
    _below_sym = _norm_case(30.0, 0.0)
    _burning = _norm_case(5.0, 200.0, burn_in=10 ** 6)
    _fam.savings, _fam.permanent_income = _saved[0], _saved[1]
    for _k, _m in _fam.members.items():
        _m.money = _saved[2][_k]
    if _saved[3]:
        _bank.wallet[_fam] = _saved[3]
    _bank.balance, _bank.taxes = _saved[4], _saved[5]
    _exp_above = 10.0 + (205.0 - 60.0) / 24
    check("Wealth norm: excess liquid wealth is spent at WEALTH_ADJUSTMENT, from deposits; below target spending is "
          "cut; no norm during the burn-in",
          abs(_above[0] - _exp_above) < 1e-9 and abs(_above[0] + _above[1] - 205.0) < 1e-9
          and abs(_below_sym[0] - (10.0 - 30.0 / 24)) < 1e-9 and abs(_burning[0] - 10.0) < 1e-9,
          f"above {_above}, below {_below_sym}, burn-in {_burning}")

# Initial money: agents aged 10+ hold WEALTH_TARGET_MONTHS of income per person times their draw over its
# mean, younger ones none
from types import SimpleNamespace as _NS  # noqa: E402
_mean_draw = np.exp(3 + 0.5 ** 2 / 2)
_ags = [_NS(age=30, money=_mean_draw), _NS(age=10, money=2 * _mean_draw), _NS(age=9, money=_mean_draw)]
_saved_months = sim.PARAMS["WEALTH_TARGET_MONTHS"]
sim.PARAMS["WEALTH_TARGET_MONTHS"] = 6
sim.generator.money_from_income(_ags, 1.5)
sim.PARAMS["WEALTH_TARGET_MONTHS"] = _saved_months
check("Initial money = months × income per person × draw / mean draw, none under 10",
      abs(_ags[0].money - 9.0) < 1e-9 and abs(_ags[1].money - 18.0) < 1e-9 and _ags[2].money == 0.0,
      f"{[a.money for a in _ags]}")

_alpha = sim.PARAMS["PRODUCTIVITY_EXPONENT"]

# The permanent-income window starts full of the initial permanent income
from agents.family import Family  # noqa: E402
_nf = Family("pi_start_test")
_nf.permanent_income = 5.0
_nf.start_permanent_income()
check("Permanent-income window full of the initial value",
      list(_nf.last_permanent_income) == [5.0] * _nf.last_permanent_window, f"{list(_nf.last_permanent_income)}")

# Sales plan: output tops the stock up to demand x (1 + ratio) within capacity and buys nothing when the
# stock covers it; workers above need are excess, and the labour market sheds them, at most half the staff
_plf = next(f for f in sim.firms.values() if f.sector not in ("Construction", "Government") and not f.pool
            and len(f.employees) > 8)
_div = sim.PARAMS["PRODUCTIVITY_MAGNITUDE_DIVISOR"]
_cap = _plf.capacity(_alpha, _div)
_saved_pl = (_plf.total_quantity, _plf.last_demand, _plf.total_balance, dict(_plf.input_inventory),
             _plf.amount_produced, _plf.amount_sold, _plf.unmet_quantity, dict(_plf.employees))
_stock0 = money_stock_total(sim)
_plf.total_quantity, _plf.last_demand = 2 * _cap, 0.5 * _cap
_q_full_stock = _plf.update_product_quantity(_alpha, _div, sim.regional_market, sim.firms, sim.seed, None, 0.2)
_spent = money_stock_total(sim) - _stock0
_plf.amount_sold, _plf.unmet_quantity = 0.3 * _cap, 0.1 * _cap
_plf.decision_on_prices_production(0.0, 0.1, sim.seed_np, sim.avg_prices, _alpha, _div, plan=0.2)
_need = 0.4 * _cap * 1.2
_exp_excess = int((_cap - _need) / (_cap / len(_plf.employees)))
_got_excess = _plf.workers_excess
_n0 = len(_plf.employees)
_lm_firms = {_plf.id: _plf}
_plf.increase_production, _plf.profit, _plf.months_unpaid, _plf.total_balance = False, 1.0, 0, 100.0
sim.labor_market.hire_fire(_lm_firms, 1.0, shed_excess=True)
_shed = _n0 - len(_plf.employees)
for _k, _e in _saved_pl[7].items():
    if _k not in _plf.employees:
        _plf.add_employee(_e)
(_plf.total_quantity, _plf.last_demand, _plf.total_balance, _inv, _plf.amount_produced, _plf.amount_sold,
 _plf.unmet_quantity) = _saved_pl[:7]
_plf.input_inventory.update(_inv)
check("Sales plan: no output and no purchase when the stock covers demand x (1 + ratio); excess workers "
      "above need; excess shed, at most half the staff",
      _q_full_stock == 0 and _plf.last_produced == 0 and abs(_spent) < 1e-9 and _got_excess == _exp_excess > 0
      and _shed == min(_exp_excess, max(1, _n0 // 2)),
      f"q {_q_full_stock}, spent {_spent}, excess {_got_excess} vs {_exp_excess}, shed {_shed}")

# SECTOR_SHARES 'ibge12': sectors that are the same CNAE sections in both classifications keep their RAIS share, the
# four regrouped ones keep their total; capacity is scaled by the sector factor, builders keep 1
from world.firms import set_sector_productivity, SECTOR_PRODUCTIVITY as _SP
_old = pd.read_csv('input/CONCURBs_SECTOR.csv', sep=';', decimal=',')
_old = _old.pivot(index='concurb_name', columns='sector', values='participation').fillna(0.0)
_old = _old.div(_old.sum(axis=1), axis=0)
_new = pd.read_csv('input/sector_shares_ibge12.csv', sep=';').pivot(index='concurb_name', columns='sector',
                                                                     values='participation').loc[_old.index]
_same = ['Agriculture', 'Mining', 'Manufacturing', 'Utilities', 'Construction', 'Transport', 'Financial', 'RealEstate']
_regrouped = ['Trade', 'Business', 'OtherServices', 'Government']
_saved_ss = sim.PARAMS['SECTOR_SHARES']
sim.PARAMS['SECTOR_SHARES'] = 'ibge12'
_gen_shares = sim.generator.sector_shares()
sim.PARAMS['SECTOR_SHARES'] = _saved_ss
_acp = sim.geo.processing_acps[0]
check("SECTOR_SHARES 'ibge12': shares sum to 1, unchanged sections keep their RAIS share, regrouped total kept, "
      "generator reads them",
      np.allclose(_new.sum(axis=1), 1, atol=1e-5) and np.allclose(_new[_same], _old[_same], atol=1e-5)
      and np.allclose(_new[_regrouped].sum(axis=1), _old[_regrouped].sum(axis=1), atol=1e-5)
      and np.isclose(_gen_shares['Business'], _new.loc[_acp, 'Business'] / _new.loc[_acp].sum()),
      f"max diff same {(_new[_same] - _old[_same]).abs().max().max():.2e}")
_fin = next(f for f in sim.firms.values() if f.employees and f.sector not in ('Construction', 'Government'))
_bld = next(f for f in sim.firms.values() if f.sector == 'Construction')
_sp_saved = (_fin.sector_productivity, _bld.sector_productivity)
_fin.sector_productivity = _bld.sector_productivity = 1.0
_cap0 = _fin.capacity(_alpha, _div)
set_sector_productivity(sim, [_fin, _bld])
_ratio = _fin.capacity(_alpha, _div) / _cap0
_bld_factor = _bld.sector_productivity
_fin.sector_productivity, _bld.sector_productivity = _sp_saved
check("Sector productivity: capacity x sector factor, builders 1",
      np.isclose(_ratio, _SP[_fin.sector]) and _bld_factor == 1.0,
      f"{_fin.sector} ratio {_ratio} vs {_SP[_fin.sector]}, builder {_bld_factor}")

# SECTOR_SHARES 'census': Census employee shares sum to 1 in every ACP, in the sectors of 'ibge12', the generator reads
# them
_cen = pd.read_csv('input/sector_shares_census.csv', sep=';').pivot(index='concurb_name', columns='sector',
                                                                     values='participation')
_saved_ss = sim.PARAMS['SECTOR_SHARES']
sim.PARAMS['SECTOR_SHARES'] = 'census'
_gen_cen = sim.generator.sector_shares()
sim.PARAMS['SECTOR_SHARES'] = _saved_ss
check("SECTOR_SHARES 'census': shares sum to 1, the sectors of 'ibge12', generator reads them",
      np.allclose(_cen.sum(axis=1), 1, atol=1e-5) and set(_cen.columns) == set(_new.columns)
      and np.isclose(_gen_cen['Construction'], _cen.loc[_acp, 'Construction'] / _cen.loc[_acp].sum()),
      f"sum range {_cen.sum(axis=1).min():.6f}-{_cen.sum(axis=1).max():.6f}, columns {sorted(_cen.columns)}")

# CONSTRUCTION_PLAN 'sales': a builder's planned demand is its goods sold plus the stock last month's houses used,
# without the money of house sales; its stock target adds its cheapest pending house; 'pipeline' reads amount_sold
from agents.firm import ConstructionFirm, plans_sales  # noqa: E402
_saved_cp = (ConstructionFirm.planned, _bld.amount_sold, _bld.house_sales, _bld.house_materials,
             _bld.last_house_materials, _bld.building, _bld.total_balance, _bld.revenue, _bld.input_cost,
             _bld.last_demand, _bld.unmet_quantity, _bld.demand_by_buyer)
_bld.amount_sold, _bld.house_sales, _bld.house_materials = 30.0, 0.0, 0.0
_bld.update_balance(500.0)
_bld.last_house_materials = 7.0
_bld.building = {0: {'cost': 40.0}, 1: {'cost': 25.0}}
ConstructionFirm.planned = False
_pipe = (_bld.goods_sold(), _bld.plan_reserve(), plans_sales(_bld))
ConstructionFirm.planned = True
_plan = (_bld.goods_sold(), _bld.plan_reserve(), plans_sales(_bld))
_bld.house_materials = 12.0
_bld.reset_amount_sold()
_rolled = (_bld.last_house_materials, _bld.house_materials, _bld.house_sales)
(ConstructionFirm.planned, _bld.amount_sold, _bld.house_sales, _bld.house_materials, _bld.last_house_materials,
 _bld.building, _bld.total_balance, _bld.revenue, _bld.input_cost, _bld.last_demand, _bld.unmet_quantity,
 _bld.demand_by_buyer) = _saved_cp
check("CONSTRUCTION_PLAN 'sales': builder demand = goods sold + last month's house stock, house money out, reserve = "
      "cheapest pending house; 'pipeline' unchanged",
      _pipe == (530.0, 0.0, False) and _plan == (37.0, 25.0, True) and _rolled == (12.0, 0.0, 0.0),
      f"pipeline {_pipe}, sales {_plan}, rolled {_rolled}")

# Social transfers: per municipality round(b x A) RGPS to the oldest, BPC to those without RGPS (65+ first), Bolsa
# Família to the poorest families; what members receive equals the ledger inflow, and it enters permanent income
from world.social_transfers import SocialTransfers
_tr = SocialTransfers(sim.mun_to_regions, sim.PARAMS['REAIS_PER_MONEY_UNIT'])
_tr_money = {a.id: a.money for a in sim.agents.values()}
_tr_led = sim.ledger['social_transfers']
_tr_paid = _tr.pay(sim)
_tr_recv = sum(a.money - _tr_money[a.id] for a in sim.agents.values())
_tr_ok, _tr_detail = True, ''
for _mun in {a.family.region_id[:7] for a in sim.agents.values() if a.family is not None and a.family.region_id}:
    _ag = [a for a in sim.agents.values() if a.family is not None and a.family.region_id
           and a.family.region_id[:7] == _mun]
    _rate, _amt = _tr.rates[_mun]['rgps']
    _rn = round(_rate * len(_ag))
    if _rn:
        _age_n = sorted((a.age for a in _ag), reverse=True)[_rn - 1]
        if any(a.last_transfer < _amt - 1e-12 for a in _ag if a.age > _age_n):
            _tr_ok, _tr_detail = False, f'{_mun} RGPS not the oldest'
    _fams = {a.family.id: a.family for a in _ag}
    _prate, _pamt = _tr.rates[_mun]['pbf']
    _pn = round(_prate * len(_ag))
    _inc = sorted(SocialTransfers.family_income(f) for f in _fams.values())
    _got = [f for f in _fams.values() if sum(m.last_transfer for m in f.members.values()) >= _pamt - 1e-9]
    if _pn and _got and max(SocialTransfers.family_income(f) for f in sorted(
            _got, key=SocialTransfers.family_income)[:_pn]) > _inc[min(_pn, len(_inc)) - 1] + 1e-12:
        _tr_ok, _tr_detail = False, f'{_mun} Bolsa Família not the poorest'
_tr_fam = next(f for f in sim.families.values() if f.members)
_tr_member = next(iter(_tr_fam.members.values()))
_tr_deque, _tr_pi = list(_tr_fam.last_permanent_income), _tr_fam.permanent_income
_tr_member_t = _tr_member.last_transfer
_tr_member.last_transfer = 0.0
_tr_fam.update_permanent_income(sim.central, sim.central.interest)
_tr_pi0 = _tr_fam.last_permanent_income[-1]
_tr_fam.last_permanent_income.clear(); _tr_fam.last_permanent_income.extend(_tr_deque)
_tr_member.last_transfer = 2.5
_tr_fam.update_permanent_income(sim.central, sim.central.interest)
_tr_pi1 = _tr_fam.last_permanent_income[-1]
_tr_fam.last_permanent_income.clear(); _tr_fam.last_permanent_income.extend(_tr_deque)
_tr_fam.permanent_income = _tr_pi
for _a in sim.agents.values():
    _a.money = _tr_money[_a.id]
    _a.last_transfer = 0.0
_tr_ledger_ok = np.isclose(sim.ledger['social_transfers'] - _tr_led, _tr_paid) and np.isclose(_tr_recv, _tr_paid)
sim.ledger['social_transfers'] = _tr_led
check("Social transfers: RGPS to the oldest, Bolsa Família to the poorest families, paid = received = ledger, "
      "part of permanent income",
      _tr_ok and _tr_ledger_ok and _tr_paid > 0 and np.isclose(_tr_pi1 - _tr_pi0, 2.5),
      f"{_tr_detail} paid {_tr_paid:.3f} received {_tr_recv:.3f}, PI step {_tr_pi1 - _tr_pi0:.3f}")

# Productivity level: the divisor makes the private staff's capacity value added (national input coefficients) equal
# the IBGE market value added per resident times the residents, net of own-account income, a month, in model money
from world.firms import set_productivity_level
_pl_saved = sim.PARAMS['PRODUCTIVITY_MAGNITUDE_DIVISOR']
_pl_div = set_productivity_level(sim)
_pl_va = pd.read_csv('input/municipal_va_2010.csv', sep=';').set_index('cod_mun')
_pl_res = [a for a in sim.agents.values() if a.family is not None and a.family.region_id
           and int(a.family.region_id[:7]) in _pl_va.index]
_pl_m = sorted({int(a.family.region_id[:7]) for a in _pl_res} | {int(m) for m in sim.mun_to_regions if int(m) in _pl_va.index})
_pl_target = (_pl_va.loc[_pl_m].va_market.sum() / _pl_va.loc[_pl_m, 'pop'].sum() * len(_pl_res) / 12
              / sim.PARAMS['REAIS_PER_MONEY_UNIT']) * (1 - sim.regional_market.pools.mixed_share)
_pl_vs = 1 - pd.read_csv('input/technical_matrix.csv').set_index('sector').sum(axis=0)
_pl_cap = sum(f.total_qualification(sim.PARAMS['PRODUCTIVITY_EXPONENT']) / _pl_div * f.sector_productivity
              * _pl_vs[f.sector] for f in sim.firms.values() if f.sector != 'Government' and not f.pool)
sim.PARAMS['PRODUCTIVITY_MAGNITUDE_DIVISOR'] = _pl_saved
check("Productivity level: capacity value added matches IBGE municipal VA net of own-account income",
      np.isclose(_pl_cap, _pl_target) and _pl_div > 0,
      f"divisor {_pl_div:.4f}, capacity VA {_pl_cap:.1f} vs {_pl_target:.1f}")

# Family wage: wage_paid is zeroed before the payroll, so a member without a job adds nothing while the paid staff add
# this month's wage
_fw_jobless = next((a for a in sim.agents.values() if a.firm_id is None and a.last_wage and a.family is not None), None)
_fw_firm = next(f for f in sim.firms.values() if f.sector != 'Government' and f.employees and f.revenue > f.input_cost)
_fw_saved = {a.id: a.wage_paid for a in sim.agents.values()}
for _a in sim.agents.values():
    _a.wage_paid = 0.0
_fw_firm.make_payment(sim.regions, sim.stats.global_unemployment_rate, sim.PARAMS['PRODUCTIVITY_EXPONENT'],
                      sim.PARAMS['TAX_LABOR'], sim.PARAMS['RELEVANCE_UNEMPLOYMENT_SALARIES'])
_fw_paid = all(e.wage_paid == e.last_wage > 0 for e in _fw_firm.employees.values())
_fw_zero = _fw_jobless is None or sum(m.wage_paid for m in _fw_jobless.family.members.values()
                                      if m.firm_id is None) == 0 and _fw_jobless.last_wage > 0
for _a in sim.agents.values():
    _a.wage_paid = _fw_saved[_a.id]
check("Family wage counts only this month's payroll",
      _fw_paid and _fw_zero, f"paid {_fw_paid}, jobless zero {_fw_zero}")

# Wage share: private firms pay the sector's national accounts share of value added, net of the own-account pool's
# share, whatever unemployment is; Government keeps its own rule
from agents.firm import Firm
_ws_firm = next(f for f in sim.firms.values() if f.sector not in ('Government', 'Construction') and not f.pool
                and f.employees and f.revenue > f.input_cost)
_ws_gov = next(f for f in sim.firms.values() if f.sector == 'Government' and f.employees)
_ws_r = sim.PARAMS['RELEVANCE_UNEMPLOYMENT_SALARIES']
_ws_tru = pd.read_csv('input/firm_income_2015.csv', sep=';').set_index('sector').wage_share.to_dict()
_ws_exp = sim.own_account.firm_wage_shares(_ws_tru)[_ws_firm.sector]
_ws_new = all(np.isclose(_ws_firm.wage_base(u, _ws_r) * _ws_firm.num_employees,
                         (_ws_firm.revenue - _ws_firm.input_cost) * _ws_exp) for u in (0.02, 0.3))
_ws_gov_same = np.isclose(_ws_gov.wage_base(0.02, _ws_r), _ws_gov.wage_base(0.3, _ws_r))
check("Wage share: the sector's share of value added net of own-account income at any unemployment; Government "
      "pays its budget's wage",
      _ws_new and _ws_exp < _ws_tru[_ws_firm.sector] + 1e-12 and _ws_gov_same,
      f"share {_ws_exp:.3f} (TRU {_ws_tru[_ws_firm.sector]:.3f}), gov same {_ws_gov_same}")

# Payout: every private firm ends at its buffer, the investment rate of what left goes to the
# investment fund and the rest out through money_profits_out; the fund is spent on FBCF products (imports through
# money_imports) and what finds no stock stays; an entrant's capital comes from outside (money_firm_entry), the
# incumbents untouched. The money stock moves exactly with the ledger throughout.
from analysis.money import money_stock_total
from world.firms import pay_out_national, fund_entrant, capital_need
_po_saved = (sim.investment_rate, sim.investment_fund, dict(sim.ledger))
_po_pe, _po_pd = sim.PARAMS['PRODUCTIVITY_EXPONENT'], sim.PARAMS['PRODUCTIVITY_MAGNITUDE_DIVISOR']
sim.investment_rate, sim.investment_fund = 0.403, 0.0
_po_m0, _po_l0 = money_stock_total(sim), sum(sim.ledger.values())
_po_out0 = sim.ledger['profits_out']
pay_out_national(sim)
_po_paid = sim.profit_share_paid
_po_at_buffer = all((f.free_cash() if f.sector == 'Construction' else f.total_balance)
                    <= capital_need(sim, f.sector, f.capacity_value(_po_pe, _po_pd)) + 1e-9
                    for f in sim.firms.values() if f.sector != 'Government' and not f.pool)
_po_split = (np.isclose(sim.investment_fund, 0.403 * _po_paid)
             and np.isclose(_po_out0 - sim.ledger['profits_out'], 0.597 * _po_paid))
_po_cons1 = np.isclose(money_stock_total(sim) - _po_m0, sum(sim.ledger.values()) - _po_l0)
_po_fund = sim.investment_fund
_po_imp0 = sim.ledger['imports']
sim.regional_market.firm_investment()
_po_spent = sim.regional_market.monthly_investment
_po_cons2 = (np.isclose(_po_spent + sim.investment_fund, _po_fund)
             and np.isclose(money_stock_total(sim) - _po_m0, sum(sim.ledger.values()) - _po_l0))
_po_bal = {f.id: f.total_balance for f in sim.firms.values()}
_po_entry0 = sim.ledger['firm_entry']
_po_new = fund_entrant(sim, next(iter(sim.regions.values())))
_po_entry = (_po_new is not None and np.isclose(sim.ledger['firm_entry'] - _po_entry0, _po_new.total_balance)
             and all(sim.firms[i].total_balance == b for i, b in _po_bal.items())
             and np.isclose(money_stock_total(sim) - _po_m0, sum(sim.ledger.values()) - _po_l0))
if _po_new is not None:
    del sim.firms[_po_new.id]
sim.investment_rate, sim.investment_fund = _po_saved[:2]
check("Payout: firms end at their buffer, investment rate to the fund and the rest out, the fund spent "
      "on FBCF with stock and ledger in step, entrants funded from outside",
      _po_paid > 0 and _po_at_buffer and _po_split and _po_cons1 and _po_cons2 and _po_spent > 0 and _po_entry,
      f"paid {_po_paid:.2f}, buffer {_po_at_buffer}, split {_po_split}, ledger {_po_cons1}/{_po_cons2}, "
      f"spent {_po_spent:.2f} of {_po_fund:.2f}, imports {_po_imp0 - sim.ledger['imports']:.2f}, entry {_po_entry}")

# Public headcount: the residence-based Census file, restricted to the run's municipalities
_gh_saved = sim.PARAMS.get('GOV_HEADCOUNT', 'pnad')
sim.PARAMS['GOV_HEADCOUNT'] = 'census'
_gh_census = sim.labor_market.process_gov_employees_year()
sim.PARAMS['GOV_HEADCOUNT'] = _gh_saved
_gh_file = pd.read_csv('input/gov_headcount_census.csv')
_gh_muns = {int(str(c)[:6]) for c in sim.geo.mun_codes}
_gh_ok = (set(_gh_census.codemun) <= _gh_muns
          and np.isclose(_gh_census[_gh_census.ano == 2010].qtde_vinc_ativos.sum(),
                         _gh_file[_gh_file.codemun.isin(_gh_muns) & (_gh_file.ano == 2010)].qtde_vinc_ativos.sum())
          and _gh_census[_gh_census.ano == 2010].qtde_vinc_ativos.sum() > 0)
check("Public headcount reads the Census file for the run's municipalities", _gh_ok,
      f"2010 census {_gh_census[_gh_census.ano == 2010].qtde_vinc_ativos.sum():.0f}")

# GOV_HEADCOUNT 'pnad': the state-ratio file, same municipalities and RAIS path, a different 2010 level
sim.PARAMS['GOV_HEADCOUNT'] = 'pnad'
_gh_pnad = sim.labor_market.process_gov_employees_year()
sim.PARAMS['GOV_HEADCOUNT'] = _gh_saved
_gh_p10 = _gh_pnad[_gh_pnad.ano == 2010].set_index('codemun').qtde_vinc_ativos
_gh_c10 = _gh_census[_gh_census.ano == 2010].set_index('codemun').qtde_vinc_ativos
_gh_growth = lambda d: d.groupby('ano').qtde_vinc_ativos.sum()
check("GOV_HEADCOUNT 'pnad' reads its file for the same municipalities with the same growth path",
      set(_gh_pnad.codemun) == set(_gh_census.codemun) and _gh_p10.sum() > 0 and not np.isclose(_gh_p10.sum(), _gh_c10.sum())
      and np.allclose(_gh_growth(_gh_pnad) / _gh_p10.sum(), _gh_growth(_gh_census) / _gh_c10.sum(), rtol=1e-3),
      f"2010 pnad {_gh_p10.sum():.0f} vs census {_gh_c10.sum():.0f}")


# Education: levels drawn per agent for its age group match the Census mix of the run's municipalities at
# 18-69; under 25 the final level is held from the school completion ages; immigrants keep it, newborns draw it
from world.education import Education, attained, YEARS
_ed = Education(sim.geo.mun_codes, np.random.RandomState(3))
_ed_ad = [a for a in sim.agents.values() if 18 <= a.age < 70]
_ed_lv = pd.Series([_ed.draw_level(str(a.region_id), a.age) for a in _ed_ad for _ in range(5)]).value_counts(normalize=True)
_ed_c = pd.read_csv('input/education_age_2010.csv', sep=';')
_ed_c = _ed_c[_ed_c.cod_mun.isin([int(m) for m in sim.geo.mun_codes]) & _ed_c.age_group.between(18, 60)]
_ed_c = _ed_c.groupby('level')['pop'].sum() / _ed_c['pop'].sum()
_ed_mix = all(abs(_ed_lv.get(l, 0) - _ed_c[l]) < 0.04 for l in YEARS)
_ed_ramp = (attained(15, 10) == 2 and attained(15, 16) == 8 and attained(15, 19) == 11 and attained(15, 22) == 15
            and attained(6, 30) == 6 and attained(2, 18) == 2)
_ed_gen = sim.generator.education
sim.generator.education = _ed
_ed_mother = next(a for a in sim.agents.values() if a.gender.lower() == 'female' and 18 <= a.age < 45)
_ed_baby = birth(sim, _ed_mother)
_ed_targets = {_a.id: _a.target for _a in sim.agents.values()}
for _a in sim.agents.values():
    _a.target = 13
_ed_clone = sim.generator.create_random_agents(1)
sim.generator.education = _ed_gen
for _a in sim.agents.values():
    _a.target = _ed_targets[_a.id]
check("Education: 18-69 level mix within 0.04 of the Census, school-age ramp, newborn and immigrant targets",
      _ed_mix and _ed_ramp and _ed_baby.target in sum(YEARS.values(), []) and _ed_baby.qualification <= 2
      and all(a.target == 13 for a in _ed_clone.values()),
      f"model {_ed_lv.sort_index().round(3).tolist()}, census {_ed_c.round(3).tolist()}, ramp {_ed_ramp}")

# Gender labels: a generated man takes male mortality and no fertility and newborns are labelled as generated
import world.demographics as _dm
_gl_man = next(a for a in sim.agents.values() if a.gender == 'male' and 20 <= a.age < 40)
_gl_woman = next(a for a in sim.agents.values() if a.gender == 'female' and 20 <= a.age < 40)
_gl_die, _gl_preg = _dm.die, _dm.pregnant
_gl_out = {}
for _gl in ['lower']:
    _gl_dead, _gl_mothers = [], []
    _dm.die = lambda s, a: _gl_dead.append(a.id)
    _dm.pregnant = lambda s, a, p: _gl_mothers.append(a.id)
    _gl_keep = [(a, a.age, a.qualification, a.p_marriage) for a in (_gl_man, _gl_woman)]
    _dm.check_demographics(sim, {30: [_gl_man, _gl_woman]}, 2010, {31: {'2010': 1.0}}, {31: {'2010': 0.0}},
                           {31: {'2010': 1.0}})
    for _a, _age, _q, _pm in _gl_keep:
        _a.age, _a.qualification, _a.p_marriage = _age, _q, _pm
    _gl_out[_gl] = (_gl_dead, _gl_mothers, _dm.birth(sim, _gl_woman).gender.islower())
_dm.die, _dm.pregnant = _gl_die, _gl_preg
check("Gender labels: generated men take male mortality, no fertility, lowercase newborns",
      _gl_out['lower'] == ([_gl_man.id], [_gl_woman.id], True), f"{_gl_out}")

# Immigration: a municipality's immigrants are offered only its vacant houses and its excess is removed from its
# residents
import world.population as _pp
_im_pops = sim.mun_pops
_im_target = (sim.pop_start, sim.pop_growth)
_im_m = min(_im_pops, key=_im_pops.get)
_im_rm = sim.housing.rental.rental_market
_im_offered = []
sim.housing.rental.rental_market = lambda fams, s, to_rent=None: _im_offered.append(
    None if to_rent is None else {h.region_id[:7] for h in to_rent})
_im_out = {}
for _im_mode, _im_delta in (('municipal', 120), ('municipal', -5)):
    sim.pop_start, sim.pop_growth = {_im_m: _im_pops[_im_m] + _im_delta}, {_im_m: 1.0}
    sim.mun_pops = defaultdict(int, {_im_m: _im_pops[_im_m]})
    _im_before = {i: a.family.region_id[:7] for i, a in sim.agents.items()}
    _pp.immigration(sim)
    _im_gone = [m for i, m in _im_before.items() if i not in sim.agents]
    _im_out[(_im_mode, _im_delta)] = (_im_offered.pop() if _im_offered else 'none', len(_im_gone),
                                      set(_im_gone) <= {_im_m})
    _im_pops[_im_m] = sim.mun_pops[_im_m]
sim.housing.rental.rental_market = _im_rm
sim.mun_pops = _im_pops
sim.pop_start, sim.pop_growth = _im_target
check("Immigration offers immigrants the municipality's vacant houses only and removes its excess from its residents",
      _im_out[('municipal', 120)][0] in (set(), {_im_m})
      and _im_out[('municipal', -5)][1] >= 5 and _im_out[('municipal', -5)][2], f"{_im_m}: {_im_out}")

# Population target: the start population grown at the 2010-2022 Census rate
_pt_days, _pt_target = sim.clock.days, (sim.pop_start, sim.pop_growth)
_pt_m = max(sim.mun_pops, key=sim.mun_pops.get)
_pt_c = pd.read_csv(_pp.CENSUS_POPULATION, sep=';', index_col='cod_mun').loc[int(_pt_m)]
sim.pop_start, sim.pop_growth = {_pt_m: 1000}, _pp.census_growth([_pt_m])
sim.clock.days = sim.PARAMS['STARTING_DAY']
_pt_0 = _pp.target_population(sim, _pt_m)
sim.clock.days = sim.PARAMS['STARTING_DAY'] + _dt.timedelta(days=round(12 * 365.25))
_pt_12 = _pp.target_population(sim, _pt_m)
sim.clock.days = _pt_days
sim.pop_start, sim.pop_growth = _pt_target
check("Population target starts at the start population and reaches it x Census 2022 / 2010 after 12 years",
      abs(_pt_0 - 1000) < 1e-9 and abs(_pt_12 / 1000 - _pt_c.pop_2022 / _pt_c.pop_2010) < 1e-6,
      f"{_pt_m}: {_pt_0}, {_pt_12}")

# Rounding of the Census cells: a region's agents add up to its Census total at the run's scale, each cell its exact value
# rounded down or up
_pr_r = next(iter(sim.regions))
_pr_pct = sim.PARAMS['PERCENTAGE_ACTUAL_POP']
_pr_c = _pp.region_counts(sim.pops, _pr_r, _pr_pct)
_pr_exact = {}
for _pr_g in ('male', 'female'):
    _pr_m = sim.pops[_pr_g][sim.pops[_pr_g]['code'] == str(_pr_r)]
    if _pr_m.empty:
        _pr_m = sim.pops[_pr_g][sim.pops[_pr_g]['code'] == str(_pr_r)[:7]]
    for _pr_a in range(101):
        _pr_col = _pr_a if _pr_a in _pr_m.columns else str(_pr_a)
        _pr_exact[(_pr_g, _pr_a)] = float(_pr_m[_pr_col].iloc[0]) * _pr_pct
_pr_near = sum(_pp.pop_age_data(sim.pops[g], _pr_r, a, _pr_pct) for g, a in _pr_exact)
check("Rounding of the Census cells keeps a region's total and rounds each cell down or up",
      sum(_pr_c.values()) == round(sum(_pr_exact.values()))
      and all(int(v) <= _pr_c[k] <= int(v) + 1 for k, v in _pr_exact.items()),
      f"{_pr_r}: remainder {sum(_pr_c.values())}, exact {sum(_pr_exact.values()):.1f}, nearest {_pr_near}")

# Car-ownership wage deciles leave out agents without a job or a wage
from types import SimpleNamespace as _cd_ns
from markets.labor import car_wage_deciles as _cd
_cd_s = [_cd_ns(last_wage=0, firm_id=None)] * 50 + [_cd_ns(last_wage=w, firm_id=1) for w in range(1, 51)]
_cd_emp = _cd(_cd_s)
check("Car-ownership deciles are taken over paid workers only",
      _cd_emp[0] > 5 and _cd_emp[-1] == 50, f"employed {_cd_emp[:3]}")

# Vale-transporte: the employer pays a transit commuter the fare above 6 % of the gross wage, a car owner nothing
from agents.firm import Firm as _VTFirm, VT_WAGE_SHARE as _vt_share
_vt_firm = next(f for f in sim.firms.values() if f.sector not in ('Government', 'Construction') and not f.own_account
                and f.num_employees >= 2)
_vt_a, _vt_b = list(_vt_firm.employees.values())[:2]
_vt_keep = [(a, a.has_car, a.commute_cost_units, a.money) for a in (_vt_a, _vt_b)]
_vt_a.has_car, _vt_b.has_car = False, True
_vt_a.commute_cost_units = _vt_b.commute_cost_units = 1e6
_vt_old = _VTFirm.vale_transporte
_vt_fare = sim.PARAMS['PUBLIC_TRANSIT_COST']
_vt_bal = _vt_firm.total_balance = 1e9
_vt_firm.revenue = max(_vt_firm.revenue, 100.0)
_vt_money = {a: a.money for a in (_vt_a, _vt_b)}
_VTFirm.vale_transporte = _vt_fare
_vt_firm.make_payment(sim.regions, 0.1, sim.PARAMS['PRODUCTIVITY_EXPONENT'], sim.PARAMS['TAX_LABOR'], 0.0)
_vt_tax = sim.PARAMS['TAX_LABOR']
_vt_gross_a = _vt_a.wage_paid / (1 - _vt_tax)
_vt_sub_a = max(0.0, 1e6 * _vt_fare - _vt_share * _vt_gross_a)
_vt_ok_pay = (abs(_vt_a.money - _vt_money[_vt_a] - _vt_a.wage_paid - _vt_sub_a) < 1e-6
              and abs(_vt_b.money - _vt_money[_vt_b] - _vt_b.wage_paid) < 1e-6
              and abs(_vt_bal - _vt_firm.total_balance - _vt_firm.wages_paid - _vt_sub_a) < 1e-3)
_VTFirm.vale_transporte = _vt_old
for _a, _car, _units, _m in _vt_keep:
    _a.has_car, _a.commute_cost_units, _a.money = _car, _units, _m
check("Vale-transporte: transit commuter paid the fare above 6 % of the gross wage, car owner nothing, firm pays it",
      _vt_ok_pay and _vt_sub_a > 0, f"pay {_vt_ok_pay} sub {_vt_sub_a:.3g}")

# POSTING_EDUCATION 'census': a vacancy is filled only from applicants of its drawn level
from world.own_account import level as _pe_level, posting_education as _pe_mix
_pe_lm = sim.labor_market
_pe_firm = next(f for f in sim.firms.values() if f.sector == 'Trade' and not f.own_account)
_pe_jobless = [a for a in sim.agents.values() if a.firm_id is None and 16 < a.age < 70 and a.family is not None
               and a.family.house is not None]
for _a in _pe_jobless:
    _a.has_car = False
_pe_low = [a for a in _pe_jobless if _pe_level(a) == 1][:5]
_pe_high = [a for a in _pe_jobless if _pe_level(a) == 4][:5]
_pe_old = sim.posting_education
sim.posting_education = {None: ([1], np.array([1.0]))}
_pe_lm.candidates = _pe_low + _pe_high
_pe_lm.matching_firm_offers([(_pe_firm, 1.0)], sim.PARAMS)
_pe_hired = [a for a in _pe_low + _pe_high if a.firm_id == _pe_firm.id]
for _a in _pe_hired:
    _pe_firm.obit(_a)
    _a.firm_id = None
_pe_lm.candidates = list(_pe_high)
_pe_lm.matching_firm_offers([(_pe_firm, 1.0)], sim.PARAMS)
_pe_none = [a for a in _pe_high if a.firm_id == _pe_firm.id]
sim.posting_education = _pe_old
_pe_lm.candidates = []
_pe_census = _pe_mix(sim.mun_to_regions)
check("POSTING_EDUCATION 'census': vacancy filled from its level only, open when none applies; mixes sum to 1",
      len(_pe_hired) == 1 and _pe_level(_pe_hired[0]) == 1 and not _pe_none and None in _pe_census
      and all(abs(p.sum() - 1) < 1e-9 for _, p in _pe_census.values()),
      f"hired {[_pe_level(a) for a in _pe_hired]}, none-level hires {len(_pe_none)}")

# Public investment unchanged during the base window, then its base real level at this month's
# price, the difference booked as a public transfer from outside
_gs_f = sim.funds
_gs_price, _gs_ledger, _gs_ext = sim.avg_prices, sim.ledger['public_transfers'], _gs_f.external_public_funding
_gs_months = _gs_f.gov_spending_months.pop('test', None)
sim.avg_prices = 2.0
_gs_n = sim.PARAMS['GOV_PAY_BURN_IN'] + sim.PARAMS['GOV_PAY_BASE_MONTHS']
_gs_seen = [_gs_f.real_public_spending('test', 10.0 if i < sim.PARAMS['GOV_PAY_BURN_IN'] else 4.0) for i in range(_gs_n)]
sim.avg_prices = 3.0
_gs_after = _gs_f.real_public_spending('test', 1.0)
_gs_ok = (_gs_seen[0] == 10.0 and _gs_seen[-1] == 4.0 and abs(_gs_after - 2.0 * 3.0) < 1e-12
          and abs(sim.ledger['public_transfers'] - _gs_ledger - 5.0) < 1e-12
          and abs(_gs_f.external_public_funding - _gs_ext - 5.0) < 1e-12)
sim.avg_prices, sim.ledger['public_transfers'], _gs_f.external_public_funding = _gs_price, _gs_ledger, _gs_ext
_gs_f.gov_spending_months.pop('test', None)
_gs_f.gov_spending_base.pop('test', None)
check("Public investment: unchanged in the base window, then the base real level at this month's price, the "
      "difference from outside", _gs_ok, f"seen {_gs_seen[0]}, {_gs_seen[-1]}, after {_gs_after}")

# Wage split: without a profile weights are qualification ** alpha; the profile adds the age profile and a persistent
# earnings factor drawn per agent and run, firms still pay exactly their wage bill, and the start keeps each area's
# income total
from agents import Agent as _ws_Agent
_ws_alpha = sim.PARAMS['PRODUCTIVITY_EXPONENT']
_ws_old = _ws_Agent.wage_profile
_ws_firm = next(f for f in sim.firms.values() if f.sector == 'Trade' and not f.own_account and f.num_employees >= 2)
_ws_staff = list(_ws_firm.employees.values())
_ws_Agent.wage_profile = None
_ws_off = all(a.wage_weight(_ws_alpha) == a.qualification ** _ws_alpha and a.wage_factor() == 1.0 for a in _ws_staff)
_ws_keep = [(a, a.earnings) for a in sim.agents.values()]
_ws_Agent.wage_profile = (0.06, -0.0006, 0.6, 12345)
for _a in _ws_staff:
    _a.earnings = None
_ws_w1 = [a.wage_weight(_ws_alpha) for a in _ws_staff]
_ws_w2 = [a.wage_weight(_ws_alpha) for a in _ws_staff]
_ws_e = _ws_staff[0].earnings
_ws_staff[0].earnings = None
_ws_redraw = _ws_staff[0].wage_factor() and _ws_staff[0].earnings == _ws_e
_ws_spread = len({round(a.earnings, 12) for a in _ws_staff}) == len(_ws_staff)
_ws_money = [(a, a.money, a.last_wage, a.wage_paid) for a in _ws_staff]
_ws_bal, _ws_rev = _ws_firm.total_balance, _ws_firm.revenue
_ws_firm.revenue = _ws_firm.total_balance = 1000.0
_ws_firm.make_payment(sim.regions, 0.05, _ws_alpha, 0.0, 0.0)
_ws_gross = sum(a.wage_paid for a in _ws_staff)
_ws_split = all(abs(a.wage_paid / _ws_gross - w / sum(_ws_w1)) < 1e-9 for (a, _, _, _), w in zip(_ws_money, _ws_w1))
_ws_bill = abs(_ws_gross - _ws_firm.wages_paid) < 1e-9
for a, m, lw, wp in _ws_money:
    a.money, a.last_wage, a.wage_paid = m, lw, wp
_ws_firm.total_balance, _ws_firm.revenue = _ws_bal, _ws_rev
_ws_pi = {f.id: f.permanent_income for f in sim.families.values()}
_ws_tot = defaultdict(float)
for f in sim.families.values():
    if f.region_id is not None:
        _ws_tot[f.region_id] += f.permanent_income
sim.initial_income_by_weight()
_ws_tot2 = defaultdict(float)
for f in sim.families.values():
    if f.region_id is not None:
        _ws_tot2[f.region_id] += f.permanent_income
_ws_start = all(abs(_ws_tot2[r] - t) <= 1e-9 * max(1.0, abs(t)) for r, t in _ws_tot.items())
for f in sim.families.values():
    f.permanent_income = _ws_pi[f.id]
for a, e in _ws_keep:
    a.earnings = e
_ws_Agent.wage_profile = _ws_old
_ws_profile = sim.wage_profile()
check("Wage split: no profile = q ** alpha; weights fixed per agent and seed, firm pays its bill in the "
      "weights, start keeps area totals, Census row loads",
      _ws_off and _ws_w1 == _ws_w2 and bool(_ws_redraw) and _ws_spread and _ws_split and _ws_bill and _ws_start
      and len(_ws_profile) == 4 and _ws_profile[2] > 0,
      f"off {_ws_off} same {_ws_w1 == _ws_w2} redraw {bool(_ws_redraw)} spread {_ws_spread} split {_ws_split} "
      f"bill {_ws_bill} start {_ws_start} profile {_ws_profile}")

# Partner matching: a partner's level is drawn from the Census spouses of the other's level, the nearest
# level when none is left; the start keeps every adult once and the first adult of each family; marriages pair
# disjoint agents from the candidates
import numpy as _fm_np
from world.family_matching import SpouseEducation as _fm_SE
from world.own_account import level as _fm_level
from world.population import census_pairs as _fm_pairs
_fm_se = _fm_SE(sim.geo.processing_acps, _fm_np.random.RandomState(7))
_fm_ad = [a for a in sim.agents.values() if a.age >= 25][:400]
_fm_by = defaultdict(list)
for _a in _fm_ad:
    _fm_by[_fm_level(_a)].append(_a)
_fm_lv1 = next(a for a in _fm_ad if _fm_level(a) == 1)
_fm_se.p[1] = _fm_np.array([0.0, 0.0, 0.0, 1.0])
_fm_got = _fm_se.pick(_fm_lv1, {4: [_fm_by[4][0]], 2: [_fm_by[2][0]]})
_fm_near = _fm_se.pick(_fm_lv1, {2: [_fm_by[2][0]], 1: [_fm_by[1][0]]})
_fm_ok_pick = _fm_level(_fm_got) == 4 and _fm_level(_fm_near) == 2
_fm_gen = sim.generator
_fm_old_sp = _fm_gen.spouses
_fm_gen.spouses = _fm_SE(sim.geo.processing_acps, _fm_np.random.RandomState(8))
_fm_fams = list(range(150))
_fm_out = _fm_gen.match_partners(list(_fm_ad), _fm_fams)
_fm_ok_start = (sorted(map(id, _fm_out)) == sorted(map(id, _fm_ad)) and _fm_out[:150] == _fm_ad[:150])
_fm_pr = _fm_pairs(sim, list(_fm_ad[:60]))
_fm_flat = [id(x) for p in _fm_pr for x in p]
_fm_ok_pairs = len(_fm_pr) == 30 and len(set(_fm_flat)) == 60 and set(_fm_flat) <= set(map(id, _fm_ad[:60]))
_fm_gen.spouses = _fm_old_sp
check("Partner matching: drawn level taken, nearest when absent; start keeps adults and heads; marriages "
      "pair disjoint candidates", _fm_ok_pick and _fm_ok_start and _fm_ok_pairs,
      f"pick {_fm_ok_pick} start {_fm_ok_start} pairs {_fm_ok_pairs}")

# House values: rents keep the INITIAL_RENTAL_PRICE level at the FipeZAP yield; building cost per m² is the state's Sinapi at
# quality 2, the CUB low / high ratios at 1 / 4, halfway at 3; a builder's planned house costs that money in output at
# its price, plus land at LOT_COST of the value
import copy as _hv_copy
from agents import House as _hv_House
from agents.firm import ConstructionFirm as _hv_CF
from world.house_values import HouseValues as _hv_HV, UF as _hv_UF
_hv = _hv_HV(sim.PARAMS)
_hv_low, _hv_high = _hv.standards[1][0], _hv.standards[1][2]
_hv_rid = next(iter(sim.regions))
_hv_unit = _hv.sinapi[_hv_UF[int(_hv_rid[:2])]] / sim.PARAMS['REAIS_PER_MONEY_UNIT']
_hv_ok_level = (abs(_hv.price_scale * _hv.rent_ratio - sim.PARAMS['INITIAL_RENTAL_PRICE']) < 1e-12
                and abs(sim.rent_ratio - _hv.rent_ratio) < 1e-12
                and abs(_hv_House.price_scale - _hv.price_scale) < 1e-12)
_hv_ok_cost = (abs(_hv.cost_per_m2(_hv_rid, 2) - _hv_unit) < 1e-12
               and abs(_hv.cost_per_m2(_hv_rid, 1) - _hv_low * _hv_unit) < 1e-12
               and abs(_hv.cost_per_m2(_hv_rid, 4) - _hv_high * _hv_unit) < 1e-12
               and abs(_hv.cost_per_m2(_hv_rid, 3) - (1 + _hv_high) / 2 * _hv_unit) < 1e-12
               and 0.7 < _hv_low < 1 < _hv_high < 1.5
               and abs(_hv.build_cost(_hv_rid, 50, 2, _hv.mean_productivity) - 50 * _hv_unit) < 1e-12)
_hv_b = next(f for f in sim.firms.values() if isinstance(f, _hv_CF) and f.prices > 0)
_hv_b2 = _hv_copy.copy(_hv_b)
_hv_b2.building, _hv_b2.houses_for_sale, _hv_b2.monthly_planned_revenue = defaultdict(dict), [], []
_hv_b2.cash_flow, _hv_b2.land_schedule, _hv_b2.total_balance = {}, None, 1e9
_hv_reg = _hv_copy.copy(sim.regions[_hv_rid])
_hv_reg.licenses, _hv_reg.treasure = 1, defaultdict(float)
_hv.sinapi = {k: 1.0 for k in _hv.sinapi}
_hv_old = sim.house_values, _hv_House.price_scale
sim.house_values, _hv_House.price_scale = _hv, _hv.price_scale
_hv_b2.plan_house([_hv_reg], sim.PARAMS, sim, np.random.RandomState(3), 0)
sim.house_values, _hv_House.price_scale = _hv_old
_hv_plan = next(iter(_hv_b2.building.values()), None)
_hv_ok_plan = (_hv_plan is not None and abs(
    _hv_plan['cost'] * _hv_b2.prices - _hv.build_cost(_hv_rid, _hv_plan['size'], _hv_plan['quality'],
                                                      _hv_b2.productivity)) < 1e-9
               and abs(1e9 - _hv_b2.total_balance - _hv_plan['quality'] * _hv_reg.index * _hv_plan['size']
                       * _hv.price_scale * sim.PARAMS['LOT_COST']) < 1e-6)
check("House values: rent level kept, cost by Sinapi state and CUB standards, builder plans in money",
      _hv_ok_level and _hv_ok_cost and _hv_ok_plan,
      f"level {_hv_ok_level} cost {_hv_ok_cost} plan {_hv_ok_plan}")

# Marriage: the population counters follow every agent who moves to another household
def _mp_gap():
    actual = defaultdict(int)
    for a in sim.agents.values():
        actual[a.family.region_id] += 1
    return {r: sim.reg_pops[r] - actual[r] for r in set(actual) | set(sim.reg_pops)}


_mp_before, _mp_mun = _mp_gap(), dict(sim.mun_pops)
_mp_check = sim.PARAMS['MARRIAGE_CHECK_PROBABILITY']
_mp_p = {a.id: a.p_marriage for a in sim.agents.values()}
sim.PARAMS['MARRIAGE_CHECK_PROBABILITY'] = 1
for a in sim.agents.values():
    a.p_marriage = 1 if a.age >= 21 else 0
_mp_regions = {i: a.family.region_id for i, a in sim.agents.items()}
_pp.marriage(sim)
_mp_moved = sum(a.family.region_id != _mp_regions[i] for i, a in sim.agents.items())
sim.PARAMS['MARRIAGE_CHECK_PROBABILITY'] = _mp_check
for a in sim.agents.values():
    a.p_marriage = _mp_p[a.id]
_mp_after = _mp_gap()
check("Marriage: population counters follow the agents who move",
      _mp_moved > 0 and all(_mp_after[r] == _mp_before.get(r, 0) for r in _mp_after)
      and sum(sim.mun_pops.values()) == sum(_mp_mun.values()),
      f"moved {_mp_moved}, gap changed in "
      f"{sum(_mp_after[r] != _mp_before.get(r, 0) for r in _mp_after)} regions")

# MARRIAGE 'census': yearly rates read as monthly probabilities
_un_rates = _pp.UnionRates()
_un_w = next(a for a in sim.agents.values() if a.gender.lower() == 'female' and 25 <= a.age < 30)
_un_t = pd.read_csv('input/union_rates_2010.csv', sep=';').set_index(['sex', 'age'])
check("Unions: monthly probabilities from the yearly Census and Registro Civil rates",
      abs(_un_rates.p(_un_w, 'formation') - (1 - (1 - _un_t.loc[('female', 25), 'formation']) ** (1 / 12))) < 1e-12
      and abs(_un_rates.p(_un_w, 'separation') - (1 - (1 - _un_t.loc[('female', 25), 'separation']) ** (1 / 12)))
      < 1e-12 and _un_rates.p(next(a for a in sim.agents.values() if a.age < 15), 'formation') == 0)

# MARRIAGE 'census': the second adult is a partner of the other sex, nearest in age, at the Census couple share
_un_adults = sorted((a for a in sim.agents.values() if a.age > 21), key=lambda a: a.id)
_un_fams = list(sim.families.values())[:len(_un_adults) // 2]
_un_mode = sim.PARAMS['MARRIAGE']
sim.PARAMS['MARRIAGE'] = 'census'
for a in _un_adults:
    a.partner = None
_un_order = sim.generator.match_partners(list(_un_adults), _un_fams)
_un_heads = _un_order[:len(_un_fams)]
_un_linked = [h for h in _un_heads if h.partner is not None]
_un_share = len(_un_linked) / len(_un_heads)
_un_gap = float(np.median([abs(h.age - h.partner.age) for h in _un_linked])) if _un_linked else np.inf
check("Unions: start couples of the other sex, near in age, at the Census couple share",
      all(h.partner.partner is h and h.gender.lower() != h.partner.gender.lower() for h in _un_linked)
      and abs(_un_share - sim.generator.couple_share) < 0.1 and _un_gap <= 3,
      f"share {_un_share:.2f} vs {sim.generator.couple_share:.2f}, median age gap {_un_gap}")
for a in _un_adults:
    a.partner = None

# MARRIAGE 'census': a separating man leaves with half the savings; the woman keeps the children and the house
_un_fam = next((f for f in sim.families.values() if f.house is not None
                and sorted(m.gender.lower() for m in f.members.values() if m.age > 21) == ['female', 'male']
                and any(m.age < 18 for m in f.members.values())), None)
# The rental market lets only family-owned vacant houses: one is handed to another family, as a sale would
_un_vacant = [h for h in sim.houses.values() if h.family_id is None and h.family_owner]
if _un_fam is not None and not _un_vacant:
    _un_h = next((h for h in sim.houses.values() if h.family_id is None and h.owner_id in sim.firms), None)
    _un_owner = next((f for f in sim.families.values() if f is not _un_fam), None)
    if _un_h is not None and _un_owner is not None:
        if _un_h in sim.firms[_un_h.owner_id].houses_for_sale:
            sim.firms[_un_h.owner_id].houses_for_sale.remove(_un_h)
        _un_h.owner_id, _un_h.family_owner = _un_owner.id, True
        _un_owner.owned_houses.append(_un_h)
        _un_vacant = [_un_h]
if _un_fam is not None and _un_vacant:
    _un_woman = next(m for m in _un_fam.members.values() if m.age > 21 and m.gender.lower() == 'female')
    _un_man = next(m for m in _un_fam.members.values() if m.age > 21 and m.gender.lower() == 'male')
    _un_woman.partner, _un_man.partner = _un_man, _un_woman
    _un_fam.savings, _un_fam.bank_savings = 10.0, 0.0
    if sim.central.wallet.get(_un_fam):
        _un_fam.savings += sim.central.withdraw(_un_fam, sim.clock.year, sim.clock.months)
    _un_kids, _un_house, _un_sav = [m for m in _un_fam.members.values() if m.age < 18], _un_fam.house, _un_fam.savings
    _mp_before = _mp_gap()
    _pp.separate(sim, _un_woman)
    _un_new = _un_man.family
    check("Unions: separation sends the man out with half the savings, the children and house stay with the woman",
          _un_new is not _un_fam and _un_new.house is not None and _un_fam.house is _un_house
          and all(k.family is _un_fam for k in _un_kids) and _un_woman.partner is None and _un_man.partner is None
          and abs(_un_fam.savings + _un_new.savings - _un_sav) < 1e-9
          and (_un_new.house.owner_id == _un_new.id or abs(_un_fam.savings - _un_sav / 2) < 1e-9)
          and all(_mp_gap()[r] == _mp_before.get(r, 0) for r in _mp_gap()),
          f"new {_un_new is not _un_fam}, savings {_un_fam.savings:.3f} + {_un_new.savings:.3f} of {_un_sav:.3f}")
else:
    check("Unions: separation (no couple with children or no vacant house to test it)", False)

# MARRIAGE 'census': unions pair women and men not in a union; a death leaves the partner single
_un_p = _pp.UnionRates.p
_pp.UnionRates.p = lambda self, agent, kind: (agent.age >= 18) * (kind == 'formation')
sim.union_rates = None
_un_before = {a.id: a.partner for a in sim.agents.values()}
_mp_before = _mp_gap()
_pp.unions(sim)
_pp.UnionRates.p = _un_p
_un_pairs = [a for a in sim.agents.values() if a.partner is not None and _un_before[a.id] is None]
check("Unions: new couples are a woman and a man not in a union before, sharing a household",
      len(_un_pairs) > 0 and all(a.partner.partner is a and a.gender.lower() != a.partner.gender.lower()
                                 and a.family is a.partner.family for a in _un_pairs)
      and all(_mp_gap()[r] == _mp_before.get(r, 0) for r in _mp_gap()), f"{len(_un_pairs)} partnered")
_un_dead = _un_pairs[0]
_un_alive = _un_dead.partner
sim.demographics.die(sim, _un_dead)
check("Unions: a death leaves the partner single", _un_alive.partner is None and _un_dead.partner is None)
sim.PARAMS['MARRIAGE'] = _un_mode

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'─' * 50}")
print(f"Results: {PASS} PASS  |  {FAIL} FAIL  |  {PASS + FAIL} total")
if FAIL:
    raise SystemExit(1)
