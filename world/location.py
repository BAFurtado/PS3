"""LOCATION_CHOICE 'logit': families choose a house among the affordable houses of their sample by
U(h) = S(h) - TC(family, region(h)) + e, e ~ Gumbel(0, LOCATION_CHOICE_SCALE x the sample's mean price x rent ratio).
LOCATION_CHOICE_SCALE 0: no draw, the house of highest S - TC.

S(h): the monthly housing-service value (model money).
    LOCATION_VALUE 'price': house price times the rent ratio.
    LOCATION_VALUE 'intrinsic': (V(h) - (price - V(h))) times the rent ratio. V(h) is the price before the
    time-on-market and vacancy terms (size x quality x the region's index, with the neighbourhood effect). The
    difference between the price and V(h) counts as a cost.
TC(family, r): the monthly time cost of the family's commutes from region r, summed over members aged 16-70:
    a member in a job: its minutes from r to its firm's region (car minutes with a private vehicle);
    an active member without a job (any adult of a family without a house yet): the job-weighted mean of its minutes
    from r to all regions;
    others: 0.
    Minutes are valued at VOT_WAGE_SHARE of the salary per hour, the salary over HOURS_PER_MONTH, and counted for
    TRIPS_PER_MONTH one-way trips. A job seeker's salary is the mean salary of the employed.
"""
from collections import defaultdict

import numpy as np

# The monthly hours behind the hourly wage (CLT, 44 h a week: divisor 220)
HOURS_PER_MONTH = 220.0
# One-way commutes a month, as in ENMU's fare-burden indicator (50 fares)
TRIPS_PER_MONTH = 50.0


class LocationChoice:
    def __init__(self, sim):
        self.sim = sim
        self.vot = sim.PARAMS['VOT_WAGE_SHARE']
        self.scale = sim.PARAMS['LOCATION_CHOICE_SCALE']
        self.intrinsic = sim.PARAMS.get('LOCATION_VALUE', 'price') == 'intrinsic'
        self.ready = False
        self.mean_salary = 0.0
        self.mean_minutes = {}

    def salary(self, agent):
        firm = self.sim.firms.get(agent.firm_id)
        if firm is None or not agent.last_wage or agent.last_wage <= 0:
            return 0.0
        return agent.last_wage * (1.0 if firm.pool else self.sim.salary_factor)

    def prepare(self):
        """Monthly: mean salary of the employed and job-weighted mean minutes from each region, by mode"""
        sim = self.sim
        transport = sim.transport
        jobs = defaultdict(int)
        salaries = []
        for a in sim.agents.values():
            if a.firm_id is not None and a.firm_id in sim.firms:
                jobs[sim.firms[a.firm_id].region_id] += 1
                s = self.salary(a)
                if s > 0:
                    salaries.append(s)
        self.mean_salary = float(np.mean(salaries)) if salaries else 0.0
        total = sum(jobs.values())
        self.mean_minutes = {}
        for car in (False, True):
            for r in sim.regions:
                self.mean_minutes[(r, car)] = (sum(n * transport.minutes_between(r, s, car) for s, n in jobs.items())
                                               / total) if total else 0.0
        self.ready = True

    def minute_value(self, salary):
        return self.vot * salary / HOURS_PER_MONTH / 60.0 * TRIPS_PER_MONTH

    def family_cost(self, family, region_id):
        sim = self.sim
        cost = 0.0
        for a in family.members.values():
            if a.is_minor or a.is_retired:
                continue
            firm = sim.firms.get(a.firm_id) if a.firm_id is not None else None
            if firm is not None:
                minutes = sim.transport.minutes_between(region_id, firm.region_id, a.has_car)
                cost += minutes * self.minute_value(self.salary(a))
            elif a.region_id is None or sim.participation.is_active(a):
                # A family without a house yet (immigrants) counts its adults as job seekers
                cost += self.mean_minutes.get((region_id, a.has_car), 0.0) * self.minute_value(self.mean_salary)
        return cost

    def service(self, house):
        if self.intrinsic:
            return (2.0 * getattr(house, 'intrinsic', house.price) - house.price) * self.sim.rent_ratio
        return house.price * self.sim.rent_ratio

    def utility(self, family, house):
        return self.service(house) - self.family_cost(family, house.region_id)

    def choose(self, family, houses):
        """The house of `houses` with the highest utility, plus a Gumbel draw when LOCATION_CHOICE_SCALE > 0"""
        if len(houses) == 1:
            return houses[0]
        costs = {}
        u = np.empty(len(houses))
        for i, h in enumerate(houses):
            if h.region_id not in costs:
                costs[h.region_id] = self.family_cost(family, h.region_id)
            u[i] = self.service(h) - costs[h.region_id]
        if self.scale > 0:
            mean_service = float(np.mean([h.price for h in houses])) * self.sim.rent_ratio
            u += self.sim.seed_np.gumbel(0.0, self.scale * mean_service, len(houses))
        return houses[int(np.argmax(u))]

    def commute_need(self, family):
        """The family's time cost from its current house as a share of its permanent income, at most 1"""
        income = family.get_permanent_income()
        if family.house is None or income <= 0:
            return 0.0
        return min(1.0, self.family_cost(family, family.house.region_id) / income)


def regional_vacancy(houses):
    """Vacancy per region, each region's rate shrunk toward its municipality's by empirical Bayes:
    v = (n v_r + k v_m) / (n + k), k = v_m (1 - v_m) / tau2, tau2 the between-region variance of vacancy in the
    municipality net of its binomial noise. tau2 = 0 gives the municipal rate."""
    counts = defaultdict(lambda: [0, 0])
    for h in houses:
        c = counts[h.region_id]
        c[0] += 1
        c[1] += 0 if h.is_occupied else 1
    by_mun = defaultdict(list)
    for r, (n, x) in counts.items():
        by_mun[str(r)[:7]].append((r, n, x))
    out = {}
    for regions in by_mun.values():
        n_tot = sum(n for _, n, _ in regions)
        p = sum(x for _, _, x in regions) / n_tot
        if len(regions) == 1 or p <= 0 or p >= 1:
            out.update({r: p for r, _, _ in regions})
            continue
        observed = sum(n * (x / n - p) ** 2 for _, n, x in regions) / n_tot
        tau2 = observed - len(regions) * p * (1 - p) / n_tot
        if tau2 <= 0:
            out.update({r: p for r, _, _ in regions})
            continue
        k = p * (1 - p) / tau2
        out.update({r: (x + k * p) / (n + k) for r, n, x in regions})
    return out
