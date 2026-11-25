import itertools
from collections import defaultdict
from math import ceil

import numpy as np
import pandas as pd

FLOWS = 'input/labour_flows_2010.csv'
FLOWS_LEVEL = 'input/labour_flows_level_2010.csv'
TRANSITIONS = 'input/pme_sector_transitions_2010.csv'
# ON_THE_JOB_SEARCH: commute time valued at half the gross wage rate (Small 2012, Valuation of travel time, Economics
# of Transportation, p.5), two trips on 22 working days a month, the wage rate the monthly wage over 220 hours (the
# CLT monthly divisor)
VALUE_OF_TIME = 0.5
TRIPS_PER_MONTH = 44
HOURS_PER_MONTH = 220


def labour_flows(by_level=False):
    """LABOUR_FLOWS 'data': the monthly employee -> unemployed and unemployed -> employed rates of the six PME
    metropolitan regions, 2010 (input/labour_flows_2010.csv, auxiliary/labour_flows.py). by_level (LABOUR_FLOWS
    'education'): {Census education level: rate} of each (input/labour_flows_level_2010.csv)"""
    if by_level:
        t = pd.read_csv(FLOWS_LEVEL, sep=';').set_index(['flow', 'level']).rate
        return ({int(k): float(v) for k, v in t['ee_u'].items()}, {int(k): float(v) for k, v in t['u_e'].items()})
    t = pd.read_csv(FLOWS, sep=';')
    t = t[t.region == 'all'].set_index('flow').rate
    return float(t['ee_u']), float(t['u_e'])


def job_search(flow='search'):
    """ON_THE_JOB_SEARCH 'pme': {Census education level: monthly rate of employees}, PME 2010
    (input/labour_flows_level_2010.csv, auxiliary/labour_flows.py): 'search', employees who took a step to get another
    job in the last 30 days, or 'j2j', job-to-job moves"""
    t = pd.read_csv(FLOWS_LEVEL, sep=';').set_index(['flow', 'level']).rate
    return {int(k): float(v) for k, v in t[flow].items()}



def car_wage_deciles(sample):
    """Wage deciles of a sample of agents for WAGE_TO_CAR_OWNERSHIP_QUANTILES: the wages of agents in a job with a
    positive wage."""
    wages = [a.last_wage for a in sample
             if a.last_wage is not None and a.firm_id is not None and a.last_wage > 0]
    return np.percentile(wages, np.arange(10, 101, 10))

class LaborMarket:
    """
    This class makes the match among firms and prospective candidates.
    The two lists (of firms and candidates) are ordered.
    The firms that pay the highest base wage and the candidates that have the most qualification.
    Firms on top choose first.
    They randomly choose with a given probability either the candidate who lives the closest
    or the most qualified.
    Lists are emptied every month.
    """

    def __init__(self, sim, seed, seed_np):
        self.sim = sim
        self.seed = seed
        self.seed_np = seed_np
        self.gov_employees = self.process_gov_employees_year()
        self.available_postings = list()
        self.candidates = list()
        self.max_dist = None
        # ON_THE_JOB_SEARCH: the month's employed job seekers and the job-to-job moves by education level
        self.on_the_job = set()
        self.moves = defaultdict(int)
        self.wage_args = (0.0, 0.0)
        self.current_wage = {}
        # HIRING_RULE 'applications': {kind: {origin sector: {destination sector: share}}} and the Census sector mix
        # by level
        self.transitions = None
        self.sector_mix = None

    def compute_max_distance(self):
        centroids = [r.addresses.centroid for r in self.sim.regions.values()]
        max_dist = 0
        for c1, c2 in itertools.combinations(centroids, 2):
            dist = c1.distance(c2)
            if dist > max_dist:
                max_dist = dist
        return max_dist

    def process_gov_employees_year(self):
        employees = pd.read_csv(f"input/gov_headcount_{self.sim.PARAMS.get('GOV_HEADCOUNT', 'census')}.csv")
        geo_codes_6_digit = [int(str(_)[:6]) for _ in self.sim.geo.mun_codes]
        # Just municipalities in this run
        return employees[employees['codemun'].isin(geo_codes_6_digit)]

    def add_post(self, firm):
        self.available_postings.append(firm)

    @property
    def num_candidates(self):
        return len(self.candidates)

    def cap_postings(self, keep, needed):
        """Keeps the first `keep` postings and at most max(1, needed - keep) of the others, drawn at random"""
        rest = self.available_postings[keep:]
        n = max(1, needed - keep)
        if len(rest) > n:
            self.available_postings = self.available_postings[:keep] + self.seed.sample(rest, n)

    def reset(self):
        self.available_postings = list()
        self.candidates = list()
        self.on_the_job = set()

    def assign_post(self, unemployment, wage_deciles, params):
        """Rank positions by revenue. Make a match as workers considers mobility choices """
        pct_distance_hiring = params['PCT_DISTANCE_HIRING']
        relevance_unemployment = params['RELEVANCE_UNEMPLOYMENT_SALARIES']

        self.seed_np.shuffle(self.candidates)
        self.wage_args = (unemployment, relevance_unemployment)
        self.current_wage = {}
        if getattr(self.sim, 'car_ownership', None) is not None:
            # CAR_OWNERSHIP 'census': car status is set monthly for all agents (world/cars.py)
            pass
        elif wage_deciles is not None:
            for c in self.candidates:
                if c.last_wage:
                    for i, d in enumerate(wage_deciles):
                        if d > c.last_wage:
                            break
                    p_car = params['WAGE_TO_CAR_OWNERSHIP_QUANTILES'][i]
                    c.has_car = self.seed_np.rand() < p_car
                else:
                    c.has_car = False
        else:
            [setattr(c, 'has_car', False) for c in self.candidates]

        # If parameter of distance or qualification is ON, firms are the ones that are divided by the criteria
        # Candidates consider distance when they deduce cost of mobility from potential wage bundle
        # Division between qualification and proximity is done randomly
        self.seed_np.shuffle(self.available_postings)
        if self.transitions is not None:
            self.match_applications(params)
            self.available_postings = []
            self.candidates = []
            self.on_the_job = set()
            return
        if len(self.available_postings) >= 2:
            split = int(len(self.available_postings) * (1 - pct_distance_hiring))
            by_qual = self.available_postings[0:split]
            by_dist = self.available_postings[split:]
        else:
            # Clear as the matching path does: a list left over is appended to next month, carrying
            # duplicates and agents whose status changed in between (e.g. died) into the matching.
            self.available_postings = []
            self.candidates = []
            self.on_the_job = set()
            return

        # Choosing by qualification
        # Firms paying higher wages first
        by_qual = [(f, f.offer_wage(unemployment, relevance_unemployment)) for f in by_qual]
        by_qual.sort(key=lambda p: p[1], reverse=True)
        by_dist = [(f, f.offer_wage(unemployment, relevance_unemployment)) for f in by_dist]
        by_dist.sort(key=lambda p: p[1], reverse=True)

        # Two matching processes. 1. By qualification 2. By distance only, if candidates left
        cand_still_looking = self.matching_firm_offers(by_qual, params, cand_looking=None, flag='qualification')
        self.matching_firm_offers(by_dist, params, cand_still_looking)

        self.available_postings = []
        self.candidates = []
        self.on_the_job = set()

    def compute_scores_cobb_douglas_vectorized(self,
                                               candidates, firm, wage,
                                               qual_min, qual_max, dist_max, wage_min, wage_max,
                                               alpha, beta,
                                               private_cost, public_cost
                                               ):
        """
        candidates: lista de candidatos
        firm: firma
        wage: salário da firma
        dist_max: tempo (ou distância) atribuído a pares sem tempo de deslocamento
        *_min, *_max: limites inferiores e superiores para normalização
        alpha, beta: parâmetros Cobb-Douglas

        QUALIFICATION_SCORE 'log': the qualification term is alpha * log(years of study), the same for a candidate at
        every firm, so the greedy match over all offers pairs the higher-wage firms with the more educated candidates
        (positive assortative matching of worker and firm types: Abowd, Kramarz & Margolis 1999; Card, Heining & Kline
        2013; in Brazil, Alvarez, Benguria, Engbom & Moser 2018; with search frictions, Shimer & Smith 2000). alpha is
        set from the Census 2010 education mix of employees by sector (runner_qualification_grid.py).
        """

        gamma = 1.0 - alpha - beta

        # Lista de atributos dos candidatos
        quals = np.array([c.qualification for c in candidates])
        # Normalização
        q_norm = (quals - qual_min) / (qual_max - qual_min + 1e-6)

        # Transit cost; TRANSPORT_MONEY 'fare': the same for car owners and transit commuters
        if getattr(self.sim, "fare_mode", False):
            transit_cost = 1.0
        else:
            has_car = np.array([c.has_car for c in candidates])
            transit_cost = np.where(has_car, private_cost, public_cost)

        # Distâncias ou tempos de deslocamento
        if self.sim.transport.matrix is not None:
            transport = self.sim.transport
            if transport.car_time is None:
                commute_time = transport.commute_time
                commutes = np.array([
                    commute_time.get((c.family.house.region_id, firm.region_id), dist_max)
                    for c in candidates
                ])
            else:
                # Car owners commute in car minutes (state 'car')
                commutes = np.array([transport.minutes_between(c.family.house.region_id, firm.region_id, c.has_car)
                                     for c in candidates])
        else:
            commutes = np.array([
                c.family.house.distance_to_firm(firm)
                for c in candidates
            ])
        # Penalty: distance * cost
        commute_penalty = commutes * transit_cost
        penalty_min = np.min(commute_penalty)
        penalty_max = np.max(commute_penalty)

        t_norm = 1.0 - (commute_penalty - penalty_min) / (penalty_max - penalty_min + 1e-6)

        # Salário é fixo por firma
        w_norm = (wage - wage_min) / (wage_max - wage_min + 1e-6)
        w_norm = np.full_like(q_norm, w_norm)

        # Evita log(0) e normaliza dentro do intervalo (0.001, 1.0)
        q_norm = np.clip(q_norm, 1e-3, 1.0)
        t_norm = np.clip(t_norm, 1e-3, 1.0)
        w_norm = np.clip(w_norm, 1e-3, 1.0)

        if self.sim.PARAMS['QUALIFICATION_SCORE'] == 'log':
            q_term = np.log(np.maximum(quals, 1.0))
        else:
            q_term = np.log(q_norm)

        # Cálculo vetorizado do log-score Cobb-Douglas
        scores = alpha * q_term + beta * np.log(t_norm) + gamma * np.log(w_norm)

        return scores


    def matching_firm_offers(self, lst_firms, params, cand_looking=None, flag=None):
        # An empty still-looking list means the first pass hired everyone: fall back to the
        # full pool only when no list was passed, or the second pass re-hires the just-hired.
        if cand_looking is not None:
            candidates = cand_looking
        else:
            candidates = self.candidates
        if not candidates:
            return None
        if not lst_firms:
            # No postings in this pass: every candidate is still looking
            return candidates if flag else None
        offers = []
        done_firms = set()
        done_cands = set()
        alpha = self.sim.PARAMS['CB_QUALIFICATION']
        beta = self.sim.PARAMS['CB_COMMUTING']
        public_cost = self.sim.PARAMS['PUBLIC_TRANSIT_COST']
        private_cost = self.sim.PARAMS['PRIVATE_TRANSIT_COST']
        # This organizes a number of offers of candidates per firm, according to their own location
        # and "size" of a firm, giving by its more recent revenue level
        # Min, Max for score attributes normalization
        qual_min, qual_max = min(c.qualification for c in candidates), max(c.qualification for c in candidates)
        if self.sim.transport.matrix is not None:
            dist_max = self.sim.transport.max_time
        else:
            if self.max_dist is None:
                self.max_dist = self.compute_max_distance()
                dist_max = self.max_dist
            else:
                dist_max = self.max_dist
        wages = [w for _, w in lst_firms]
        wage_min, wage_max = min(wages), max(wages)

        education = self.sim.posting_education
        if education is not None:
            from world.own_account import level
            by_level = defaultdict(list)
            for c in candidates:
                by_level[level(c)].append(c)
        for firm, wage in lst_firms:
            pool = candidates
            if education is not None:
                # POSTING_EDUCATION 'census': the vacancy's level, from its sector's employees; open if none applies
                levels, p = education.get(firm.sector, education[None])
                pool = by_level.get(levels[self.seed_np.choice(len(levels), p=p)])
                if not pool:
                    continue
            sampled_candidates = self.seed.sample(pool, min(len(pool), int(params['HIRING_SAMPLE_SIZE'])))
            scores = self.compute_scores_cobb_douglas_vectorized(
                sampled_candidates, firm, wage,
                qual_min, qual_max, dist_max, wage_min, wage_max,
                alpha, beta,
                private_cost, public_cost
            )
            for c, s in zip(sampled_candidates, scores):
                offers.append((firm, c, s))

        # Then, the criteria is used to order all candidates
        offers = sorted(offers, key=lambda o: o[2], reverse=True)
        acceptance = getattr(self.sim, 'commute_acceptance', None)
        for firm, candidate, score in offers:
            if firm not in done_firms and candidate not in done_cands:
                employed = candidate in self.on_the_job
                if employed and (candidate.firm_id is None or candidate.firm_id == firm.id
                                 or not self.better_job(candidate, firm, wage)):
                    # ON_THE_JOB_SEARCH: an employee laid off since the search, or offered no better job, passes
                    continue
                if acceptance is not None:
                    # COMMUTE_ACCEPTANCE: a rejected offer passes to the firm's next candidate
                    home = candidate.family.house.region_id
                    minutes = self.sim.transport.minutes_between(home, firm.region_id, candidate.has_car)
                    if self.seed_np.rand() >= acceptance.probability(minutes, home, candidate.has_car):
                        continue
                if employed:
                    self.leave_for(candidate)
                self.apply_assign(candidate, firm)
                done_firms.add(firm)
                done_cands.add(candidate)

        # If this run was for qualification, another run for distance has to go through
        if flag:
            # Now it is time for the matching for firms favoring proximity
            cand_still_looking = [c for c in self.candidates if c not in done_cands]
            return cand_still_looking
        return None

    def load_applications(self, mun_codes):
        """HIRING_RULE 'applications': the PME 2010 sector moves as row shares (input/pme_sector_transitions_2010.csv)
        and the run's Census sector mix by level (world/own_account.sector_mix)"""
        from world.own_account import sector_mix
        t = pd.read_csv(TRANSITIONS, sep=';')
        self.transitions = {}
        for (kind, origin), g in t.groupby(['kind', 'origin']):
            self.transitions.setdefault(kind, {})[origin] = (g.set_index('destination').persons
                                                             / g.persons.sum()).to_dict()
        self.sector_mix = sector_mix(mun_codes)

    def application_shares(self, agent, employed):
        """{sector: share} of the agent's applications before commuting: the PME row of its last sector (j2j for an
        employee, u2e for the unemployed, the other kind for an origin without a row), else the Census sector mix of
        its level"""
        from world.own_account import level
        origin = agent.last_sector
        kinds = ('j2j', 'u2e') if employed else ('u2e', 'j2j')
        for kind in kinds:
            row = self.transitions.get(kind, {}).get(origin)
            if row is not None:
                return row
        return self.sector_mix.get(level(agent), self.sector_mix[None])

    def match_applications(self, params):
        """HIRING_RULE 'applications': applications weighted by sector moves and commute, postings ranked by their
        level and qualification, firm-proposing deferred acceptance on job value (Gale and Shapley 1962, College
        admissions and the stability of marriage, American Mathematical Monthly)"""
        from world.own_account import level
        postings = self.available_postings
        seekers = [c for c in self.candidates if not (c in self.on_the_job and c.firm_id is None)]
        if not postings or not seekers:
            return
        transport = self.sim.transport
        acceptance = self.sim.commute_acceptance
        education = self.sim.posting_education
        offers = {}
        for f in postings:
            if f.id not in offers:
                offers[f.id] = f.offer_wage(*self.wage_args)
        wage = [offers[f.id] for f in postings]
        posting_level = []
        for f in postings:
            levels, p = education.get(f.sector, education[None])
            posting_level.append(levels[self.seed_np.choice(len(levels), p=p)])
        sectors = sorted({f.sector for f in postings})
        sector_index = np.array([sectors.index(f.sector) for f in postings])
        regions = sorted({f.region_id for f in postings})
        region_index = np.array([regions.index(f.region_id) for f in postings])
        k = int(params['HIRING_SAMPLE_SIZE'])
        weights = {}
        applicants = defaultdict(list)
        value = {}
        reservation = {}
        for i, c in enumerate(seekers):
            employed = c in self.on_the_job
            home = c.family.house.region_id
            key = (home, bool(c.has_car), employed, c.last_sector, level(c) if c.last_sector is None else None)
            if key not in weights:
                shares = self.application_shares(c, employed)
                a = np.array([shares.get(s, 0.0) for s in sectors])[sector_index]
                p = np.array([acceptance.probability(transport.minutes_between(home, r, c.has_car), home, c.has_car)
                              for r in regions])[region_index]
                weights[key] = a * p
            w = weights[key]
            if employed:
                w = np.where(np.array([f.id == c.firm_id for f in postings]), 0.0, w)
                current = self.sim.firms.get(c.firm_id)
                if current is not None:
                    if current.id not in self.current_wage:
                        self.current_wage[current.id] = current.offer_wage(*self.wage_args)
                    reservation[i] = self.job_value(c, current, self.current_wage[current.id])
            n = int(np.count_nonzero(w))
            if n == 0:
                continue
            for j in self.seed_np.choice(len(postings), min(k, n), replace=False, p=w / w.sum()):
                applicants[j].append(i)
                value[(i, j)] = self.job_value(c, postings[j], wage[j])
        ranked = {}
        for j, lst in applicants.items():
            ranked[j] = sorted(lst, key=lambda i: (level(seekers[i]) == posting_level[j], seekers[i].qualification,
                                                   value[(i, j)]), reverse=True)
        held = {}
        nxt = defaultdict(int)
        free = list(ranked)
        while free:
            j = free.pop()
            lst = ranked[j]
            while nxt[j] < len(lst):
                i = lst[nxt[j]]
                nxt[j] += 1
                v = value[(i, j)]
                if i in reservation and v <= reservation[i]:
                    continue
                h = held.get(i)
                if h is None or v > h[0]:
                    held[i] = (v, j)
                    if h is not None:
                        free.append(h[1])
                    break
        for i, (_, j) in held.items():
            c = seekers[i]
            if c in self.on_the_job:
                self.leave_for(c)
            self.apply_assign(c, postings[j])

    def job_value(self, agent, firm, wage):
        """ON_THE_JOB_SEARCH: the wage less the money cost of the commute and its time, valued at VALUE_OF_TIME of
        the wage rate over TRIPS_PER_MONTH one-way trips. Under TRANSPORT_MONEY 'fare' the fare bill is the same at
        every job and only the time counts. Workers value commute against wage in job search: Manning and Petrongolo
        (2017), How local are labor markets?, American Economic Review; Le Barbanchon, Rathelot and Roulet (2021),
        Gender differences in job search: trading off commute against wage, Quarterly Journal of Economics."""
        transport = self.sim.transport
        if transport.matrix is not None:
            minutes = transport.minutes_between(agent.family.house.region_id, firm.region_id, agent.has_car)
            units = minutes * transport.cost_factor
            time = VALUE_OF_TIME * wage / HOURS_PER_MONTH * TRIPS_PER_MONTH * minutes / 60
        else:
            units, time = agent.family.house.distance_to_firm(firm), 0.0
        if getattr(self.sim, 'fare_mode', False):
            money = 0.0
        else:
            params = self.sim.PARAMS
            money = units * (params['PRIVATE_TRANSIT_COST'] if agent.has_car else params['PUBLIC_TRANSIT_COST'])
        return wage - money - time

    def better_job(self, agent, firm, wage):
        """The offer of `firm` at `wage` is worth more than the agent's current job at its firm's offered wage
        (on-the-job search with posted wages: Burdett and Mortensen 1998, Wage differentials, employer size, and
        unemployment, International Economic Review)"""
        current = self.sim.firms.get(agent.firm_id)
        if current is None:
            return True
        if current.id not in self.current_wage:
            self.current_wage[current.id] = current.offer_wage(*self.wage_args)
        return self.job_value(agent, firm, wage) > self.job_value(agent, current, self.current_wage[current.id])

    def leave_for(self, agent):
        """ON_THE_JOB_SEARCH: the agent quits its firm, which refills the post (REPLACE_SEPARATIONS)"""
        from world.own_account import level
        current = self.sim.firms.get(agent.firm_id)
        if current is not None and agent.id in current.employees:
            del current.employees[agent.id]
            current.pending_replacements += 1
        self.moves[level(agent)] += 1

    def apply_assign(self, chosen, firm):
        chosen.set_commute(firm, self.sim.transport)
        firm.add_employee(chosen)

    def look_for_jobs(self, agents):
        participation = self.sim.participation
        self.candidates += [agent for agent in agents.values()
                            if agent.firm_id is None and participation.is_active(agent)]

    def gov_target(self, sim):
        """Government headcount of the current year (RAIS) at the population's scale"""
        return ceil(self.gov_employees[self.gov_employees.ano == sim.clock.year].qtde_vinc_ativos.sum() *
                    sim.PARAMS['PERCENTAGE_ACTUAL_POP'])

    def gov_hire_fire(self, sim):
        total_gov_employees = self.gov_target(sim)
        gov_firms = [firm for firm in sim.firms.values()
                     if firm.sector == 'Government']
        total_employment = sum([f.num_employees for f in gov_firms])
        jobs_balance = total_gov_employees - total_employment

        if jobs_balance > 0:
            posts_per_firm = max(1, ceil(jobs_balance / len(gov_firms)))
            posted = 0
            for f in sim.seed.sample(gov_firms, len(gov_firms)):
                if posted >= jobs_balance:
                    break
                n = min(posts_per_firm, jobs_balance - posted)
                for _ in range(n):
                    self.add_post(f)
                posted += n
        elif jobs_balance < 0:
            to_fire = -jobs_balance
            fires_per_firm = max(1, ceil(to_fire / len(gov_firms)))
            fired = 0
            for f in sim.seed.sample(gov_firms, len(gov_firms)):
                if fired >= to_fire:
                    break
                for _ in range(min(fires_per_firm, to_fire - fired)):
                    if f.employees:
                        f.fire(self.seed)
                        fired += 1

    def hire_fire(self, firms, firm_enter_freq, initialize=False, fire_unpaid_months=0, planned_growth=False,
                  replace_separations=False, gov_headcount_only=False, shed_excess=False, startup_quota=None):
        """Firms adjust their labor force based on profit. With initialize, each firm other than Government posts one
        vacancy, and with startup_quota ({firm id: staff}) only while its staff is below its quota.
        With gov_headcount_only, Government firms are skipped.
        With planned_growth a growing firm posts as many vacancies as its production plan needs. With replace_separations, a firm that is not shrinking
        also re-posts one vacancy for each worker it lost to natural separation or death since its last adjustment.
        With shed_excess, a firm
 that is not growing sheds its workers_excess, at most half
        its staff."""
        random_value = self.seed_np.random(size=len(firms.values()))
        n_fired = 0
        # CONSTRUCTION_PLAN 'sales': builders follow the rules of the other firms
        construction_planned = shed_excess and self.sim.PARAMS.get('CONSTRUCTION_PLAN', 'pipeline') == 'sales'
        for i, firm in enumerate(firms.values()):
            if firm.own_account:
                continue
            # `firm_enter_freq` is the frequency firms enter the market
            if random_value[i] < firm_enter_freq:
                fired_before = n_fired
                if initialize:
                    if firm.sector != 'Government' and (startup_quota is None
                                                        or firm.num_employees < startup_quota.get(firm.id, 0)):
                        self.add_post(firm)
                    continue
                elif gov_headcount_only and firm.sector == 'Government':
                    # gov_hire_fire alone sets its headcount: Government profit is always negative (wages are
                    # funded by tax transfers), so the profit rule below would fire its staff every month.
                    firm.pending_replacements = 0
                    continue
                elif firm.total_balance <= 0:
                    # Insolvent: shed labour regardless of production signal
                    firm.fire(self.seed_np)
                    n_fired += 1
                elif (fire_unpaid_months > 0
                      and firm.sector != 'Government'
                      and firm.months_unpaid >= fire_unpaid_months):
                    # Could not pay its staff for fire_unpaid_months months running: shed one worker
                    # and stop posting. Short droughts (erratic sales) are tolerated, sustained ones are not.
                    # Government is excluded: gov_hire_fire sets its headcount.
                    firm.fire(self.seed_np)
                    n_fired += 1
                elif firm.sector == 'Construction' and not construction_planned:
                    # Construction barely sells into the goods market that
                    # increase_production/profit are derived from. Use signals
                    # tied directly to the house-building pipeline instead.
                    increase, oversupplied = firm.labor_signals(self.sim.PARAMS)
                    if increase:
                        self.add_growth_posts(firm, planned_growth)
                    elif oversupplied:
                        firm.fire(self.seed_np)
                        n_fired += 1
                elif firm.increase_production and firm.profit >= 0:
                    self.add_growth_posts(firm, planned_growth)
                elif shed_excess and firm.workers_excess > 0:
                    for _ in range(min(firm.workers_excess, max(1, firm.num_employees // 2))):
                        firm.fire(self.seed_np)
                        n_fired += 1
                    firm.workers_excess = 0
                elif not firm.increase_production and firm.profit < 0:
                    # Fire only when BOTH signals align: surplus inventory AND losing money.
                    # OR-logic fired profitable firms with adequate stock, collapsing demand.
                    firm.fire(self.seed_np)
                    n_fired += 1
                # Replacement: refill workers lost to natural separation or death, unless the firm is shrinking
                # (fired this adjustment, or not paying its staff). Government's headcount is set by gov_hire_fire.
                if (replace_separations and firm.sector != 'Government'
                        and n_fired == fired_before and firm.months_unpaid == 0):
                    for _ in range(firm.pending_replacements):
                        self.add_post(firm)
                firm.pending_replacements = 0
        # print('N of fired: ',n_fired)

    def add_growth_posts(self, firm, planned_growth):
        # With planned_growth, a growing firm posts the workers its production decision says it needs
        # (firm.workers_needed), at least one and at most its current headcount (no more than doubling in a month).
        # Builders plan on the house pipeline, not on goods sales, and keep one post. Otherwise one post a month.
        n = 1
        if planned_growth and firm.sector != 'Construction':
            n = max(1, min(firm.workers_needed, firm.num_employees))
        for _ in range(n):
            self.add_post(firm)

    def __repr__(self):
        return self.available_postings, self.candidates
