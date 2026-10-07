"""Own-account work (OwnAccountPools, below): the own-account workers of each sector share its own-account part of
every purchase of it."""
from collections import defaultdict

import numpy as np
import pandas as pd

from agents.firm import Firm

FILE = 'input/own_account_2010.csv'
INCOME = 'input/own_account_income_2010.csv'
LABOUR_SHARE = 'input/firm_income_2015.csv'
LEVELS = [1, 2, 3, 4]


def level(agent):
    """Census education level of the agent's years of study (world/education.py YEARS)"""
    q = agent.qualification
    return 1 if q < 4 else 2 if q < 9 else 3 if q < 12 else 4


def posting_education(mun_codes):
    """POSTING_EDUCATION 'census': the education levels of the Census 2010 employees (V6930 1-3) aged 17-69 of each
    sector in the run's municipalities (input/own_account_2010.csv), {sector: (levels, probabilities)}; key None: all
    sectors, for a sector without employees there"""
    t = pd.read_csv(FILE, sep=';')
    t = t[t.cod_mun.isin([int(m) for m in mun_codes]) & (t.position == 'employee') & (t.sector != 'Unknown')]
    mix = t.groupby(['sector', 'level']).persons.sum()
    out = {}
    for sector, m in [(None, t.groupby('level').persons.sum())] + [(s, mix[s]) for s in mix.index.levels[0]
                                                                     if s in mix.index.get_level_values(0)]:
        m = m[m > 0]
        if m.sum() > 0:
            out[sector] = (list(m.index), (m / m.sum()).to_numpy())
    return out


class OwnAccountCensus:
    """The run's Census 2010 own-account share of the employed aged 17-69 of each education level and the own-account
    sector mix of each level (input/own_account_2010.csv, auxiliary/own_account.py)"""

    def __init__(self, sim):
        t = pd.read_csv(FILE, sep=';')
        t = t[t.cod_mun.isin([int(m) for m in sim.mun_to_regions])]
        employed = t[t.position == 'employed'].groupby('level').persons.sum()
        own = t[t.position == 'own_account']
        self.share = (own.groupby('level').persons.sum() / employed).reindex(LEVELS).fillna(0.0).to_dict()
        mix = own[~own.sector.isin(['Unknown', 'Government'])].groupby(['level', 'sector']).persons.sum()
        self.sectors = {}
        for lv in LEVELS:
            m = mix.get(lv) if lv in mix.index.get_level_values(0) else None
            if m is None or m.sum() <= 0:
                m = mix.groupby('sector').sum()
            self.sectors[lv] = (list(m.index), (m / m.sum()).to_numpy())


class OwnAccountPool(Firm):
    """The own-account workers of one sector. It holds no stock and is never sampled by buyers; it
    is paid directly its sector's share (OwnAccountPools.share) of household, government, investment and input purchases
    and of exports, buys the inputs of that output on the matrix like a firm, and pays the rest to its members in the
    q^α weights."""
    pool = True
    own_account = True
    # Members' pay last month it had members (OwnAccountPools.monthly)
    last_net = 0.0

    def receive(self, amount, regions, tax_consumption, consumer_region_id, if_origin, external=False):
        """A purchase from the pool, net of consumption tax as Firm.sale (an export charged at destination leaves its
        tax outside)"""
        revenue = amount * (1 - tax_consumption)
        self.total_balance += revenue
        self.revenue += revenue
        if external and not if_origin:
            return
        region = self.region_id if if_origin else consumer_region_id
        regions[region if region in regions else self.region_id].collect_taxes(amount * tax_consumption, "consumption")

    def update_product_quantity(self, *args, **kwargs):
        self.last_capacity = 0.0
        self.last_produced = 0.0
        return 0

    def decision_on_prices_production(self, *args, **kwargs):
        self.increase_production = False
        self.workers_excess = 0

    def create_externalities(self, *args, **kwargs):
        pass

    def invest_eco_efficiency(self, *args, **kwargs):
        self.inno_inv = 0.0

    def pay_taxes(self, *args, **kwargs):
        pass

    def capacity(self, *args, **kwargs):
        return 0.0

    def capacity_value(self, *args, **kwargs):
        return 0.0

    def make_payment(self, regions, unemployment, alpha, tax_labor, relevance_unemployment, tax_transport=False):
        """Inputs for this month's output at the sector's mean price, then all cash to the members"""
        sim = self.sim
        market = sim.regional_market
        prices = [f.prices for f in sim.firms.values()
                  if f.sector == self.sector and not f.pool and f.num_employees > 0]
        price = float(np.mean(prices)) if prices else (sim.avg_prices or 1.0)
        if self.revenue > 0:
            self.buy_inputs(self.revenue / price, market, sim.firms, sim.seed, market.technical_matrix,
                            market.ext_local_matrix)
            for s in self.input_inventory:
                self.input_inventory[s] = 0.0
        self.wages_paid = 0.0
        if not self.employees or self.total_balance <= 0:
            return
        weights = {a: a.wage_weight(alpha) for a in self.employees.values()}
        total = sum(weights.values())
        for a, w in weights.items():
            pay = self.total_balance * w / total
            a.money += pay
            a.last_wage = pay
            a.wage_paid = pay
        self.wages_paid = self.total_balance
        self.total_balance = 0.0


class OwnAccountPools:
    """The pools and their members.

    share[s] = own-account workers' share of the work income of sector s in the run's municipalities (Census 2010,
    input/own_account_income_2010.csv) x the sector's labour share of value added (input/firm_income_2015.csv, TRU
    2015, mixed income included): the pool's share of the sector's value added, and so the part of each local purchase
    of s (households, government, investment, firms' inputs; after any imported part) and of its exports paid to the
    pool of s while it has members. Firms then pay (labour share - share) / (1 - share) of their value added.

    Start: the Census 2010 own-account share of the employed of each level (OwnAccountCensus) joins the pool of a
    sector drawn from its level's own-account mix. Each month (after job matching) members, then active agents without work,
    decide one at a time in random order, each with probability LABOR_MARKET: a member's pay is its q^α share of its
    pool's pay last month over the members still in it, a searcher's that share with itself counted in, in a sector
    drawn from its level's mix; the search is worth (1 - u) times the mean wage of private employees of its level
    (Harris-Todaro). A member leaves when the pool pays less, a searcher joins when it pays more; each move changes the
    pay the next one sees, so moves stop where the two are equal."""

    def __init__(self, sim):
        self.sim = sim
        self.census = OwnAccountCensus(sim)
        inc = pd.read_csv(INCOME, sep=';')
        inc = inc[inc.cod_mun.isin([int(m) for m in sim.mun_to_regions]) & ~inc.sector.isin(['Unknown', 'Government'])]
        inc = inc.groupby('sector')[['work_income', 'own_account_income']].sum()
        labour = pd.read_csv(LABOUR_SHARE, sep=';').set_index('sector').wage_share
        labour = labour.reindex(inc.index)
        self.share = (inc.own_account_income / inc.work_income * labour).fillna(0.0).to_dict()
        # Own-account income over the value added it implies (work income / labour share), all sectors
        self.mixed_share = float(inc.own_account_income.sum() / (inc.work_income / labour).sum())
        self.labour = labour.to_dict()
        mun = max(sim.mun_to_regions, key=lambda m: len(sim.mun_to_regions[m]))
        region = sim.regions[sim.mun_to_regions[mun][0]]
        sectors = sorted({s for names, _ in self.census.sectors.values() for s in names})
        address = sim.generator.get_random_points_in_polygon(region, number_addresses=len(sectors))
        self.pools = {}
        for i, s in enumerate(sectors):
            pool = OwnAccountPool(sim.generator.gen_id(), address[i], 0.0, region.id, sector=s)
            pool.sim = sim
            sim.firms[pool.id] = pool
            self.pools[s] = pool
        self.joined = self.left = 0

    def firm_wage_shares(self, wage_shares):
        """The labour share firms pay when the pool's share of value added is not theirs: (labour share - pool share) /
        (1 - pool share)"""
        out = dict(wage_shares)
        for s, sh in self.share.items():
            if s in out and 0 < sh < 1:
                out[s] = max(0.0, (out[s] - sh) / (1 - sh))
        return out

    def payable(self, sector):
        """The pool of `sector` while it has members, and the part of a purchase paid to it"""
        pool = self.pools.get(sector)
        if pool is None or not pool.employees:
            return None, 0.0
        return pool, self.share.get(sector, 0.0)

    def join(self, agent, sector=None):
        if sector is None:
            names, p = self.census.sectors[level(agent)]
            sector = names[self.sim.seed_np.choice(len(names), p=p)]
        self.pools[sector].add_employee(agent)
        agent.set_commute(None)

    def leave(self, pool, agent):
        del pool.employees[agent.id]
        agent.firm_id = None
        agent.set_commute(None)

    def start(self, candidates, nonemployment):
        by_level = defaultdict(list)
        for a in candidates:
            by_level[level(a)].append(a)
        for lv in LEVELS:
            pool = by_level[lv]
            n = int(round(self.census.share[lv] * (1 - nonemployment) * len(pool)))
            if n <= 0:
                continue
            for i in self.sim.seed_np.choice(len(pool), size=min(n, len(pool)), replace=False):
                self.join(pool[i])

    def monthly(self, unemployment):
        sim = self.sim
        alpha = sim.PARAMS['PRODUCTIVITY_EXPONENT']
        freq = sim.PARAMS['LABOR_MARKET']
        wages, factors = defaultdict(list), defaultdict(list)
        for f in sim.firms.values():
            if not f.own_account and f.sector != 'Government':
                for a in f.employees.values():
                    if a.last_wage:
                        wages[level(a)].append(a.last_wage)
                        factors[level(a)].append(a.wage_factor())
        search = {lv: (1 - unemployment) * np.mean(w) for lv, w in wages.items() if w}
        # The private pay an agent compares is its level's mean scaled by its own age and earnings factor over the
        # level's mean factor
        mean_factor = {lv: np.mean(f) for lv, f in factors.items() if f}

        def expected(a, lv):
            return search[lv] * a.wage_factor() / mean_factor[lv]

        weight = {}
        for s, pool in self.pools.items():
            if pool.wages_paid > 0:
                pool.last_net = pool.wages_paid
            weight[s] = sum(a.wage_weight(alpha) for a in pool.employees.values())

        def pay(s, q, joining):
            w = weight[s] + (q if joining else 0.0)
            return self.pools[s].last_net * q / w if w > 0 else 0.0

        # One at a time, in random order, each move changing the pay the next one sees
        members = [(p, a) for p in self.pools.values() for a in p.employees.values()]
        for i in sim.seed_np.permutation(len(members)):
            pool, a = members[i]
            lv, q = level(a), a.wage_weight(alpha)
            if lv in search and sim.seed_np.random() < freq and pay(pool.sector, q, False) < expected(a, lv):
                self.leave(pool, a)
                weight[pool.sector] -= q
                self.left += 1
        participation = sim.participation
        searchers = [a for a in sim.agents.values()
                     if a.firm_id is None and a.family is not None and a.family.house is not None
                     and participation.is_active(a)]
        for i in sim.seed_np.permutation(len(searchers)):
            agent = searchers[i]
            lv = level(agent)
            if lv not in search or sim.seed_np.random() >= freq:
                continue
            names, p = self.census.sectors[lv]
            sector = names[sim.seed_np.choice(len(names), p=p)]
            q = agent.wage_weight(alpha)
            if pay(sector, q, True) > expected(agent, lv):
                self.join(agent, sector)
                weight[sector] += q
                self.joined += 1
