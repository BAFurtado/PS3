import pandas as pd
from collections import defaultdict
from math import ceil
from itertools import chain


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

    def process_gov_employees_year(self):
        employees = pd.read_csv('input/qtde_vinc_gov_rais.csv')
        # Just municipalities in this run
        return employees[employees['codemun'].isin(self.sim.geo.mun_codes)]

    def add_post(self, firm):
        self.available_postings.append(firm)

    @property
    def num_candidates(self):
        return len(self.candidates)

    def reset(self):
        self.available_postings = list()
        self.candidates = list()

    def assign_post(self, unemployment, wage_deciles, params):
        """Rank positions by revenue. Make a match as workers considers mobility choices """
        pct_distance_hiring = params['PCT_DISTANCE_HIRING']
        relevance_unemployment = params['RELEVANCE_UNEMPLOYMENT_SALARIES']

        self.seed.shuffle(self.candidates)
        if wage_deciles is not None:
            for c in self.candidates:
                if c.last_wage:
                    for i, d in enumerate(wage_deciles):
                        if d > c.last_wage:
                            break
                    p_car = params['WAGE_TO_CAR_OWNERSHIP_QUANTILES'][i]
                    c.has_car = self.seed.random() < p_car
                else:
                    c.has_car = False
        else:
            [setattr(c, 'has_car', False) for c in self.candidates]

        # If parameter of distance or qualification is ON, firms are the ones that are divided by the criteria
        # Candidates consider distance when they deduce cost of mobility from potential wage bundle
        # Division between qualification and proximity is done randomly
        self.seed.shuffle(self.available_postings)
        if len(self.available_postings) >= 2:
            split = int(len(self.available_postings) * (1 - pct_distance_hiring))
            by_qual = self.available_postings[0:split]
            by_dist = self.available_postings[split:]
        else:
            return

        # Choosing by qualification
        # Firms paying higher wages first
        by_qual = [(f, f.wage_base(unemployment, relevance_unemployment)) for f in by_qual]
        by_qual.sort(key=lambda p: p[1], reverse=True)
        by_dist = [(f, f.wage_base(unemployment, relevance_unemployment)) for f in by_dist]
        by_dist.sort(key=lambda p: p[1], reverse=True)

        # Two matching processes. 1. By qualification 2. By distance only, if candidates left
        cand_still_looking = self.matching_firm_offers(by_qual, params, cand_looking=None, flag='qualification')
        self.matching_firm_offers(by_dist, params, cand_still_looking)

        self.available_postings = []
        self.candidates = []

    def matching_firm_offers(self, lst_firms, params, cand_looking=None, flag=None):
        if cand_looking:
            candidates = cand_looking
        else:
            candidates = self.candidates
        offers = []
        done_firms = set()
        done_cands = set()
        # This organizes a number of offers of candidates per firm, according to their own location
        # and "size" of a firm, giving by its more recent revenue level
        for firm, wage in lst_firms:
            candidates = self.seed.sample(candidates, min(len(candidates), int(params['HIRING_SAMPLE_SIZE'])))
            for c in candidates:
                transit_cost = params['PRIVATE_TRANSIT_COST'] if c.has_car else params['PUBLIC_TRANSIT_COST']
                score = wage - (c.family.house.distance_to_firm(firm) * transit_cost)
                if flag:
                    offers.append((firm, c, c.qualification + score))
                else:
                    offers.append((firm, c, score))

        # Then, the criteria is used to order all candidates
        offers = sorted(offers, key=lambda o: o[2], reverse=True)
        for firm, candidate, score in offers:
            if firm not in done_firms and candidate not in done_cands:
                self.apply_assign(candidate, firm)
                done_firms.add(firm)
                done_cands.add(candidate)

        # If this run was for qualification, another run for distance has to go through
        if flag:
            # Now it is time for the matching for firms favoring proximity
            cand_still_looking = [c for c in self.candidates if c not in done_cands]
            return cand_still_looking

    def apply_assign(self, chosen, firm):
        chosen.set_commute(firm)
        firm.add_employee(chosen)

    def look_for_jobs(self, agents):
        self.candidates += [agent for agent in agents.values() if 16 < agent.age < 70 and agent.firm_id is None]

    def gov_hire_fire(self, sim):
        total_gov_employees = ceil(self.gov_employees[self.gov_employees.ano == sim.clock.year].qtde_vinc_ativos.sum() *
                                  sim.PARAMS['PERCENTAGE_ACTUAL_POP'])
        mun_gov_firms = defaultdict(list)
        these_regions = list(chain.from_iterable(sim.mun_to_regions.values()))
        gov_firms = [firm for firm in sim.firms.values()
                     if firm.sector == 'Government']
        firms_num_employees = [f.num_employees() for f in gov_firms]
        total_employment = sum(firms_num_employees)
        jobs_balance = total_gov_employees - total_employment
        if jobs_balance > 0:
            hiring_firms = sim.seed_np.choice(gov_firms, size=jobs_balance)
            [self.add_post(f) for f in hiring_firms]
        else:
            firing_firms = sim.seed_np.choice(gov_firms, size=jobs_balance * -1)
            [f.fire(self.seed) for f in firing_firms]
        sim.funds.mun_gov_firms = mun_gov_firms

    def hire_fire(self, firms, firm_enter_freq, initialize=False):
        """Firms adjust their labor force based on profit"""
        random_value = self.seed_np.random(size=len(firms.values()))
        for i, firm in enumerate(firms.values()):
            # `firm_enter_freq` is the frequency firms enter the market
            if random_value[i] < firm_enter_freq:
                # Increase production based on low inventory and low prices
                if initialize or firm.increase_production:
                    self.add_post(firm)
                # Three-way criteria: Wages exceed sales, profits (considering taxes) are negative
                # and there is no need to increase production due to low prices and inventories
                elif firm.profit < 0 and firm.wages_paid > firm.revenue:
                    if not firm.increase_production:
                        firm.fire(self.seed)
                        # Condition is valid only once
                        firm.increase_production = True

    def __repr__(self):
        return self.available_postings, self.candidates
