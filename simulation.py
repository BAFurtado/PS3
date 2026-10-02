import copy
import datetime
import itertools
import json
import math
import os
import pickle
import random
import sys
import secrets
from collections import defaultdict

import numpy as np
import pandas as pd

import analysis
import conf
import markets
from world import Generator, demographics, clock, population
from world.firms import firm_growth, firm_exit, size_initial_capital, set_productivity_level, pay_profit_shares, pay_out_national, set_sector_productivity
from world.funds import Funds
from analysis.money import money_stock_total
from world.geography import Geography, STATES_CODES, state_string
from agents.firm import Firm, UNPLANNED_SECTORS, import_parity
from world.transport import TransportNetwork
from world.participation import Participation
from world.social_transfers import SocialTransfers
from world.own_account import OwnAccount, OwnAccountPools, posting_education
from markets.goods import RegionalMarket, External


def apply_closure(params):
    """CLOSURE 'open': the parameters named in CLOSURE_OPEN take its values; 'legacy' leaves them as given"""
    if params.get('CLOSURE', 'legacy') == 'open':
        params = {**params, **params['CLOSURE_OPEN']}
    return params


def resolve_seed(params):
    """Return the seed for this run.

    An explicit per-run `SEED` in PARAMS takes precedence: sensitivity runs give the
    same seed to every policy configuration of a replication, so a treated run and its
    baseline differ only by the policy and their difference is an exact counterfactual.
    Without one, fall back to conf.RUN — a fresh random seed per run under
    KEEP_RANDOM_SEED, otherwise the fixed conf.RUN['SEED'].
    """
    seed = params.get("SEED")
    if seed is None:
        seed = (
            secrets.randbelow(2 ** 32)
            if conf.RUN["KEEP_RANDOM_SEED"]
            else conf.RUN.get("SEED", 0)
        )
    return int(seed)


class Simulation:
    def __init__(self, params, output_path):
        params = apply_closure(params)
        self.PARAMS = copy.copy(params)
        self.geo = Geography(params, self.PARAMS["STARTING_DAY"].year)
        self.regional_market = RegionalMarket(self)
        self.clock = clock.Clock(self.PARAMS["STARTING_DAY"])
        self.output = analysis.Output(self, output_path)
        self.stats = analysis.Statistics(params)
        self.logger = analysis.Logger(hex(id(self))[-5:])
        self.funds = Funds(self)
        self._seed = resolve_seed(self.PARAMS)
        # Record the seed actually used, so a run is reproducible from its own config.
        self.PARAMS["SEED"] = self._seed
        self.seed = random.Random(self._seed)
        self.seed_np = np.random.RandomState(self._seed)
        self.generator = Generator(self)
        # Generate the external supplier
        self.avg_prices = 1
        self.external = External(self, self.PARAMS["TAXES_STRUCTURE"]["consumption_equal"])
        self.mun_pops = defaultdict(int)
        self.reg_pops = defaultdict(int)
        self.demographics = demographics
        self.grave = list()
        # Exited firms by id (FIRM_EXIT_MONTHS), with exit_date and exit_reason set
        self.firm_grave = dict()
        # Negative balances written off at exit (money already paid out that the firm did not have)
        self.firm_exit_writeoff = 0.0
        # Money crossing the ACP's boundary, cumulative by channel (analysis/money.py), and the stock it started with
        self.ledger = defaultdict(float)
        self.money_initial = 0.0
        # INITIAL_MONEY 'target': the ACP's Census income per person aged 10+ at the start, in model money
        self.income_per_person = 0.0
        # FIRM_PAYOUT: profit shares paid this month ('national': all cash above the buffers paid out)
        self.profit_share_paid = 0.0
        # FIRM_PAYOUT 'national': corporate FBCF / gross operating surplus, and the money set aside for investment
        self.investment_rate = 0.0
        self.investment_fund = 0.0
        # Entries skipped because the sector's incumbents had too little capital above their buffer
        self.firm_entry_unfunded = 0
        self.mun_to_regions = defaultdict(set)
        # PARTICIPATION 'census': who is in the labour force (world/participation.py)
        self.participation = None
        # SOCIAL_TRANSFERS 'data': federal benefits paid to residents (world/social_transfers.py)
        self.social_transfers = None
        # OWN_ACCOUNT: own-account work (world/own_account.py)
        self.own_account = None
        # POSTING_EDUCATION 'census': {sector (None: all): (levels, probabilities)} of a vacancy's education
        self.posting_education = None
        # Read necessary files — loaded as dicts for fast O(1) lookup in demographics
        self.m_men, self.m_women, self.f = dict(), dict(), dict()

        for state in self.geo.states_on_process:
            m_men = pd.read_csv(
                "input/Demografia/2_Mortality/mortality_men_%s.csv" % state,
                header=0,
                decimal=".",
            ).set_index("age")
            m_men.columns = m_men.columns.astype(str)
            self.m_men[state] = m_men.to_dict('index')

            m_women = pd.read_csv(
                "input/Demografia/2_Mortality/mortality_women_%s.csv" % state,
                header=0,
                decimal=".",
            ).set_index("age")
            m_women.columns = m_women.columns.astype(str)
            self.m_women[state] = m_women.to_dict('index')

            f = pd.read_csv(
                "input/Demografia/1_Fertility/fertility_%s.csv" % state,
                header=0,
                decimal=".",
            ).set_index("age")
            f.columns = f.columns.astype(str)
            self.f[state] = f.to_dict('index')

        # Travel-time matrix between APs, if the processing ACPs have one (world/transport.py)
        self.transport = TransportNetwork(self.PARAMS, self.geo.processing_acps, self.logger.logger)
        if self.transport.matrix is not None:
            self.transport.update(self.PARAMS['STARTING_DAY'])
        self.labor_market = markets.LaborMarket(self, self.seed, self.seed_np)
        self.housing = markets.HousingMarket()
        self.heads = population.HouseholdsHeads(self)
        self.pops, self.total_pop = population.load_pops(
            self.geo.mun_codes, self.PARAMS, self.geo.year
        )
        # Interest
        # Average interest rate - Earmarked new operations - Households - Real estate financing - Market rates
        # PORT. Taxa média de juros das operações de crédito com recursos direcionados - Pessoas físicas -
        # Financiamento imobiliário com taxas de mercado. BC series 433. 25497. 4390.
        # Values before 2011-03-01 when the series began are set at the value of 2011-03-01. After, mean.
        interest = pd.read_csv(f"input/interest_{self.PARAMS['INTEREST']}.csv")
        interest.date = pd.to_datetime(interest.date)
        self.interest = interest.set_index("date")
        # sbpe/fgts rates are regulated independently of SELIC; governed by INTEREST_HOUSING param.
        housing_key = self.PARAMS.get('INTEREST_HOUSING', 'media')
        housing_interest = pd.read_csv(f"input/planhab_funds/interest_housing_{housing_key}.csv")
        housing_interest.date = pd.to_datetime(housing_interest.date)
        self.housing_interest = housing_interest.set_index("date")

        # Subsidies configuration: convert flat value to per-sector defaultdict
        level = self.PARAMS['ECO_INVESTMENT_SUBSIDIES']
        self.PARAMS['ECO_INVESTMENT_SUBSIDIES'] = defaultdict(lambda: level)
        # Targeted sectors: zero out non-targeted sectors if enabled
        if self.PARAMS['TARGETED_SUBSIDIES']:
            sectors = pd.read_csv('input/emissions_sectors.csv', dtype={'mun_code': str}).isic_12
            for sector in sectors:
                if sector not in self.PARAMS['TARGETED_SECTORS']:
                    self.PARAMS['ECO_INVESTMENT_SUBSIDIES'][sector] = 0

    def update_pop(self, old_region_id, new_region_id):
        if old_region_id and new_region_id:
            # Agents are moving from the old to the new region
            self.mun_pops[old_region_id[:7]] -= 1
            self.reg_pops[old_region_id] -= 1
            self.mun_pops[new_region_id[:7]] += 1
            self.reg_pops[new_region_id] += 1
        elif old_region_id is None:
            # New agents are coming into the new region
            self.mun_pops[new_region_id[:7]] += 1
            self.reg_pops[new_region_id] += 1
        elif new_region_id is None:
            # Agents have died
            self.mun_pops[old_region_id[:7]] -= 1
            self.reg_pops[old_region_id] -= 1

    def generate(self):
        """Spawn or load regions, agents, houses, families, and firms"""
        save_file = "{}.agents".format(self.output.save_name)
        if not os.path.isfile(save_file) or conf.RUN["FORCE_NEW_POPULATION"]:
            self.logger.logger.info("Creating new agents")
            # Key moment when creation of agents happen!
            regions = self.generator.create_regions()
            agents, houses, families, firms = self.generator.create_all(regions)
            agents = {
                a: agents[a] for a in agents.keys() if agents[a].address is not None
            }
            with open(save_file, "wb") as f:
                pickle.dump([agents, houses, families, firms, regions], f)
        else:
            self.logger.logger.info("Loading existing agents")
            with open(save_file, "rb") as f:
                agents, houses, families, firms, regions = pickle.load(f)
            # This Generator did not mint these ids, so its counter is behind them.
            self.generator.resume_ids(agents, houses, families, firms)

        # Initialize populations directly
        for agent in agents.values():
            r_id = agent.region_id
            mun_code = r_id[:7]
            self.reg_pops[r_id] += 1
            self.mun_pops[mun_code] += 1
        return regions, agents, houses, families, firms, self.generator.central

    def run(self, log=True):
        """Runs the simulation"""
        self.logger.logger.info("Starting run.")
        self.logger.logger.info("Output: {}".format(self.output.path))
        if log:
            self.logger.logger.info(
                "Params: {}".format(json.dumps(self.PARAMS, indent=4, default=str))
            )
            self.logger.logger.info("Seed: {}".format(self._seed))

            self.logger.logger.info("Running...")
        starting_day = self.PARAMS["STARTING_DAY"]
        total_days = self.PARAMS["TOTAL_DAYS"]
        while self.clock.days < starting_day + datetime.timedelta(days=total_days):
            self.daily()
            if self.clock.months == 1 and conf.RUN["SAVE_TRANSIT_DATA"]:
                self.output.save_transit_data(self, "start")
            if self.clock.new_month:
                self.monthly()
            if self.clock.new_quarter:
                self.quarterly()
            if self.clock.new_year:
                self.yearly()
            self.clock.days += datetime.timedelta(days=1)

        if conf.RUN["PRINT_FINAL_STATISTICS_ABOUT_AGENTS"]:
            self.logger.log_outcomes(self)

        if conf.RUN["SAVE_TRANSIT_DATA"]:
            self.output.save_transit_data(self, "end")
        self.output.close()
        self.logger.logger.info("Simulation completed.")

    def initialize(self):
        """Initiating simulation"""
        self.logger.logger.info("Initializing...")

        (
            self.regions,
            self.agents,
            self.houses,
            self.families,
            self.firms,
            self.central,
        ) = self.generate()
        self.central.ledger = self.ledger
        # Also for a population loaded from file
        set_sector_productivity(self, self.firms.values())
        if self.PARAMS.get('INITIAL_MONEY', 'lognormal') == 'target':
            self.initial_money_from_income()
        if self.PARAMS.get('PI_START', 'reset') == 'census':
            for family in self.families.values():
                family.start_permanent_income()

        if self.transport.matrix is not None:
            self.transport.check_coverage(self.regions.keys())
            self.transport.calibrate_cost(self.regions, self.reg_pops)

        # Group regions into their municipalities
        for region_id in self.regions.keys():
            mun_code = region_id[:7]
            self.mun_to_regions[mun_code].add(region_id)
        # Region order feeds the sequence of random draws downstream, so it must be
        # stable across processes.
        for mun_code, regions in self.mun_to_regions.items():
            self.mun_to_regions[mun_code] = sorted(regions)
        if self.PARAMS.get('PARTICIPATION', 'off') == 'census':
            self.participation = Participation(self.mun_to_regions, self._seed)
            self.stats.participation = self.participation
        if self.PARAMS.get('SOCIAL_TRANSFERS', 'off') == 'data':
            self.social_transfers = SocialTransfers(self.mun_to_regions, self.PARAMS['REAIS_PER_MONEY_UNIT'])
        if self.PARAMS.get('FIRM_PAYOUT', 'none') == 'national':
            self.investment_rate = float(pd.read_csv('input/investment_rate_2015.csv', sep=';').investment_rate.iloc[0])
        Firm.own_account_market = self.PARAMS.get('OWN_ACCOUNT', 'off') == 'firms'
        Firm.vale_transporte = self.PARAMS['PUBLIC_TRANSIT_COST'] if self.PARAMS.get('VALE_TRANSPORTE', False) else None
        if self.PARAMS.get('POSTING_EDUCATION', 'off') == 'census':
            self.posting_education = posting_education(self.mun_to_regions)
        if Firm.own_account_market:
            self.own_account = OwnAccount(self)
        elif self.PARAMS.get('OWN_ACCOUNT', 'off') == 'pool':
            self.own_account = OwnAccountPools(self)
            self.regional_market.pools = self.own_account
        Firm.wage_shares = (pd.read_csv('input/firm_income_2015.csv', sep=';').set_index('sector').wage_share.to_dict()
                            if self.PARAMS.get('WAGE_SHARE', 'unemployment') == 'tru' else None)
        if Firm.wage_shares is not None and isinstance(self.own_account, OwnAccountPools):
            # The pool's share of value added is no longer firms'
            Firm.wage_shares = self.own_account.firm_wage_shares(Firm.wage_shares)
        elif Firm.wage_shares is not None and self.own_account is not None:
            # Own-account income is no longer part of firms' value added
            Firm.wage_shares = pd.read_csv('input/own_account_productivity_2010.csv', sep=';').set_index(
                'sector').wage_share_firms.to_dict()

        # First jobs allocated
        # Create an existing job market
        self.labor_market.look_for_jobs(self.agents)
        total = actual = self.labor_market.num_candidates
        actual_unemployment = self.stats.global_unemployment_rate
        # Share of those aged 17-69 left without a job (INITIAL_EMPLOYMENT)
        target = self.initial_nonemployment()
        census = self.PARAMS.get('INITIAL_EMPLOYMENT', 'legacy') == 'census'
        if self.own_account is not None:
            self.own_account.start(self.labor_market.candidates, target)
            self.labor_market.candidates = [c for c in self.labor_market.candidates if c.firm_id is None]
            actual = self.labor_market.num_candidates
        while actual / total > target:
            # Government is staffed to its RAIS headcount by gov_hire_fire, not by one post per firm:
            # otherwise it takes start-up hires in proportion to its firm count, and sheds the excess in month 1.
            self.labor_market.gov_hire_fire(self)
            n_gov = len(self.labor_market.available_postings)
            self.labor_market.hire_fire(self.firms, 1, initialize=True)
            if census:
                # No more posts than the jobs still missing to the target
                self.labor_market.cap_postings(n_gov, math.ceil(actual - target * total))
            self.labor_market.assign_post(actual_unemployment, None, self.PARAMS)
            self.labor_market.look_for_jobs(self.agents)
            actual = self.labor_market.num_candidates
        self.labor_market.reset()
        divisor = set_productivity_level(self)
        if self.PARAMS.get('PRODUCTIVITY_LEVEL', 'divisor') == 'municipal':
            self.logger.logger.info(f'PRODUCTIVITY_LEVEL municipal: PRODUCTIVITY_MAGNITUDE_DIVISOR {divisor:.4f}')
        size_initial_capital(self)

        for i, family in enumerate(self.families.values()):
            head_family = max(family.members.values(), key=lambda x: x.last_wage)
            head_family.set_head_family()

        # Update initial pop
        for region in self.regions.values():
            region.pop = self.reg_pops[region.id]
        self.money_initial = money_stock_total(self)
        self.central.equity_target = self.central.equity()

    def initial_nonemployment(self):
        """INITIAL_EMPLOYMENT 'census': the Census 2010 share of those aged 17-69 without a job in the run's
        municipalities; under PARTICIPATION 'census', the share of the active without a job. 'legacy': 0.086, the mean
        unemployment rate of six metropolitan regions in January 2000"""
        if self.PARAMS.get('INITIAL_EMPLOYMENT', 'legacy') != 'census':
            return 0.086
        if self.participation is not None:
            return self.participation.unemployment
        census = pd.read_csv('input/nonemployment_2010.csv', sep=';').set_index('cod_mun')
        census = census.loc[[int(m) for m in self.mun_to_regions]]
        return 1 - census.employed.sum() / census.pop_17_69.sum()

    def leave_labour_force(self):
        """PARTICIPATION 'census': employed agents no longer active leave their job, which the firm may refill"""
        for agent in self.agents.values():
            if agent.firm_id is not None and not self.participation.is_active(agent):
                firm = self.firms.get(agent.firm_id)
                if firm is not None and agent.id in firm.employees:
                    del firm.employees[agent.id]
                    firm.pending_replacements += 1
                agent.firm_id = None
                agent.set_commute(None)

    def initial_money_from_income(self):
        """INITIAL_MONEY 'target': each family's members aged 10+ hold WEALTH_TARGET_MONTHS of its Census income per
        person (its initial permanent income over them), times their own draw over the draw's mean"""
        income, adults = 0.0, 0
        for family in self.families.values():
            n = sum(1 for m in family.members.values() if m.age >= 10)
            if n:
                self.generator.money_from_income(family.members.values(), family.permanent_income / n)
                income += family.permanent_income
                adults += n
        self.income_per_person = income / adults if adults else 0.0

    def daily(self):
        pass

    def monthly(self):
        if self.transport.matrix is not None:
            self.transport.update(self.clock.days)
            # Keep commutes current with the network in force and with house moves
            for agent in self.agents.values():
                if agent.firm_id is not None and agent.family is not None:
                    firm = self.firms.get(agent.firm_id)
                    if firm is not None:
                        agent.set_commute(firm, self.transport)
        # Set interest rates
        interests = self.interest[
            self.interest.index.date == self.clock.days][['interest', 'mortgage', ]].iloc[0]
        mask = self.housing_interest.index.normalize() == pd.to_datetime(self.clock.days)
        housing_interests = self.housing_interest.loc[mask].iloc[0]

        mortgage = housing_interests['mortgage'] if 'mortgage' in housing_interests else interests['mortgage']
        values = [interests['interest'], mortgage, housing_interests['sbpe'], housing_interests['fgts']]
        self.central.set_interest(*values)

        current_unemployment = self.stats.global_unemployment_rate

        # Create new land licenses.
        # Small cities have fewer neighborhoods but proportionally more free urban land,
        # so we ensure a city-wide floor: effective rate = max(per-region param, floor/n_regions).
        licenses_per_region = self.PARAMS["EXPECTED_LICENSES_PER_REGION"]
        min_city_monthly = self.PARAMS.get("LICENSE_MIN_CITY_MONTHLY", 0)
        if min_city_monthly > 0:
            n_regions = len(self.regions)
            licenses_per_region = max(licenses_per_region, min_city_monthly / n_regions)
        for region in self.regions.values():
            region.licenses += self.seed_np.poisson(lam=licenses_per_region)

        # Firms that stayed insolvent or idle leave, then new firms enter according to average historical growth
        firm_exit(self)
        firm_growth(self)

        # Update firm products
        prod_exponent = self.PARAMS["PRODUCTIVITY_EXPONENT"]
        prod_magnitude_divisor = self.PARAMS["PRODUCTIVITY_MAGNITUDE_DIVISOR"]
        [f.reset_amount_sold() for f in self.firms.values()]
        # Build sector→firm map once per month and pass to update_product_quantity
        sector_firm_map = {}
        for f in self.firms.values():
            sector_firm_map.setdefault(f.sector, []).append(f)
        if Firm.own_account_market:
            # Staff weights of each sector's sellers this month, for input purchases (Firm.choose_firm_per_sector)
            self.regional_market.input_cum = {s: list(itertools.accumulate(max(1, f.num_employees) for f in fs))
                                              for s, fs in sector_firm_map.items()}
        # PRODUCTION_PLAN 'sales': private firms other than builders produce for last month's demand plus the stock
        # target; builders plan on the house pipeline and Government's headcount is set by its budget
        planning = self.PARAMS.get('PRODUCTION_PLAN', 'capacity') == 'sales'
        plan = self.PARAMS.get("INVENTORY_TARGET_RATIO", 0.0) if planning else None
        for firm in self.firms.values():
            firm.update_product_quantity(prod_exponent, prod_magnitude_divisor,
                                         self.regional_market,
                                         self.firms,
                                         self.seed,
                                         sector_firm_map,
                                         plan if firm.sector not in UNPLANNED_SECTORS or firm.own_account else None)

        # Call demographics
        # Update agent life cycles
        for state in self.geo.states_on_process:
            mortality_men = self.m_men[state]
            mortality_women = self.m_women[state]
            fertility = self.f[state]

            state_str = state_string(state, STATES_CODES)

            birthdays = defaultdict(list)
            for agent in self.agents.values():
                if (
                    self.clock.months == agent.month
                    and agent.region_id[:2] == state_str
                ):
                    birthdays[agent.age].append(agent)

            demographics.check_demographics(
                self,
                birthdays,
                self.clock.year,
                mortality_men,
                mortality_women,
                fertility,
            )

        # Calculate head_rate as input for immigration adjustments
        self.stats.calculate_head_rate(self.families.values(), self.clock.days.strftime("%Y-%m-%d"))

        # Adjust population for immigration
        population.immigration(self)

        # Adjust families for marriages
        population.marriage(self)

        # Firms initialization
        for firm in self.firms.values():
            firm.present = self.clock.days

        # FAMILIES CONSUMPTION -- using payment received from previous month
        for family in self.families.values():
            family.update_permanent_income(self.central, self.central.interest)
        # Equalize money within family members
        # Tax consumption when doing sales are realized
        self.regional_market.consume()
        # Government firms consumption
        self.regional_market.government_consumption()
        # FIRM_PAYOUT 'national': investment demand from last month's payout
        self.regional_market.firm_investment()
        # External consumption based on internal household and government consumption
        internal_consumption = defaultdict(float)
        for key, value in self.regional_market.monthly_gov_consumption.items():
            internal_consumption[key] += value
        for key, value in self.regional_market.monthly_hh_consumption.items():
            internal_consumption[key] += value
        self.external.final_consumption(internal_consumption, self.seed)
        # Make rent payments
        self.housing.process_monthly_rent(self)
        # Collect loan repayments
        self.central.collect_loan_payments(self)

        # FIRMS
        # Accessing dictionary parameters outside the loop for performance
        tax_labor = self.PARAMS["TAX_LABOR"]
        tax_firm = self.PARAMS["TAX_FIRM"]
        is_policy_active = self.clock.days > self.PARAMS['STARTING_DAY'] + datetime.timedelta(self.PARAMS['ECO_POLICY_DAYS'])
        tax_emission = self.PARAMS["TAX_EMISSION"] if is_policy_active else 0
        relevance_unemployment = self.PARAMS["RELEVANCE_UNEMPLOYMENT_SALARIES"]
        sticky = self.PARAMS["STICKY_PRICES"]
        markup = self.PARAMS["MARKUP"]
        const_cash_flow = self.PARAMS["CONSTRUCTION_ACC_CASH_FLOW"]
        price_ruggedness = self.PARAMS["PRICE_RUGGEDNESS"]
        inventory_target_ratio = self.PARAMS.get("INVENTORY_TARGET_RATIO", 0.0)
        price_markup_cap = self.PARAMS.get("PRICE_MARKUP_CAP", 0.25)
        demand_signal_unmet = self.PARAMS.get("DEMAND_SIGNAL_UNMET", False)
        price_demand_response = self.PARAMS.get("PRICE_DEMAND_RESPONSE", 0.0)
        tax_transport = self.PARAMS["TAX_TRANSPORT"]
        plan = inventory_target_ratio if self.PARAMS.get('PRODUCTION_PLAN', 'capacity') == 'sales' else None
        self.avg_prices, _ = self.stats.update_price(self.firms, mid_simulation_calculus=True)
        # IMPORT_PARITY_PRICING: tradable firms price against the tradable average, capped at import parity, with no
        # price response to refused demand; the others against the non-tradable average
        parity = self.PARAMS.get('IMPORT_PARITY_PRICING', False)
        if parity:
            tradables = set(self.PARAMS['TRADABLE_SECTORS'])
            avg_t, avg_n = self.stats.group_prices(self.firms, tradables)
            avg_t, avg_n = avg_t or self.avg_prices, avg_n or self.avg_prices
            parity_ceiling = import_parity(self.PARAMS)
        if self.PARAMS.get('FAMILY_WAGE', 'last') == 'month':
            for agent in self.agents.values():
                agent.wage_paid = 0.0
        for firm in self.firms.values():
            # Tax workers when paying salaries
            firm.make_payment(
                self.regions,
                current_unemployment,
                prod_exponent,
                tax_labor,
                relevance_unemployment,
                tax_transport)
            # Firms update generated externalities, based on own sector and wages paid this month
            firm.create_externalities(self.regions, tax_emission, self.PARAMS['EMISSIONS_PARAM'])
            # Tax firms before profits: (revenue - salaries paid)
            firm.pay_taxes(self.regions, tax_firm)
            # Profits are after taxes
            firm.calculate_profit()
            # Check whether it is necessary to update prices
            tradable = parity and firm.sector in tradables
            firm.decision_on_prices_production(
                sticky,
                markup,
                self.seed_np,
                (avg_t if tradable else avg_n) if parity else self.avg_prices,
                prod_exponent,
                prod_magnitude_divisor,
                const_cash_flow,
                price_ruggedness,
                inventory_target_ratio,
                price_markup_cap,
                demand_signal_unmet,
                0.0 if tradable else price_demand_response,
                parity_ceiling[firm.sector] if tradable else None,
                plan if firm.sector not in UNPLANNED_SECTORS or firm.own_account else None,
            )
            firm.invest_eco_efficiency(
                self.regional_market,
                self.regions,
                self.seed_np)

        if self.PARAMS.get('FIRM_PAYOUT', 'none') == 'staff':
            pay_profit_shares(self)
        elif self.PARAMS.get('FIRM_PAYOUT', 'none') == 'national':
            pay_out_national(self)
        if self.social_transfers is not None:
            self.social_transfers.pay(self)

        # Construction firms
        # Probability depends (strongly) on market supply
        if self.PARAMS["OFFER_SIZE_ON_PRICE"]:
            vacancy = self.stats.vacancy_rate
        else:
            vacancy = .1
        construction_firms = [f for f in self.firms.values() if f.sector == 'Construction' and not f.own_account]

        for firm in construction_firms:
            # See if firm can build a house
            firm.plan_house(
                self.regions.values(),
                self.PARAMS,
                self,
                self.seed_np,
                vacancy,
            )
            # See whether a house has been completed. If so, register. Else, continue
            house = firm.build_house(self.regions, self.generator)
            if house is not None:
                self.houses[house.id] = house

        # Initiating Labor Market
        # AGENTS
        if self.participation is not None:
            self.leave_labour_force()
        self.labor_market.look_for_jobs(self.agents)

        # FIRMS
        # Government labor first (initialization is for all firms, government specific is monthly)
        self.labor_market.gov_hire_fire(self)
        # Check if new employee needed. Check if firing is necessary
        # 3-way criteria: Wages/sales, profits, and increase production
        self.labor_market.hire_fire(self.firms, self.PARAMS["LABOR_MARKET"],
                                    fire_unpaid_months=self.PARAMS.get("FIRE_UNPAID_MONTHS", 0),
                                    planned_growth=self.PARAMS.get("PLANNED_GROWTH_POSTS", False),
                                    replace_separations=self.PARAMS.get("REPLACE_SEPARATIONS", False),
                                    gov_headcount_only=self.PARAMS.get("GOV_REVISED", False),
                                    shed_excess=self.PARAMS.get('PRODUCTION_PLAN', 'capacity') == 'sales')

        # Job Matching
        # Sample used only to calculate wage deciles
        agent_values = list(self.agents.values())
        sample_size = math.floor(len(agent_values) * 0.5)
        last_wages = [a.last_wage for a in self.seed.sample(agent_values, sample_size)
                      if a.last_wage is not None]
        del agent_values
        wage_deciles = np.percentile(last_wages, np.arange(10, 101, 10))
        self.labor_market.assign_post(current_unemployment, wage_deciles, self.PARAMS)
        if self.own_account is not None:
            self.own_account.monthly(current_unemployment)

        # Natural job separation: workers quit/reach contract end at an exogenous monthly rate.
        # Runs after matching so separated workers miss this month's pool and must wait
        # until next month — creating a minimum one-month unemployment spell per separation.
        sep_rate = self.PARAMS.get('NATURAL_SEPARATION_RATE', 0.0)
        if sep_rate > 0:
            eligible = [a for a in self.agents.values() if a.firm_id is not None and 16 < a.age < 70
                        and not self.firms[a.firm_id].own_account]
            to_separate = [a for a, s in zip(eligible, self.seed_np.random(len(eligible)) < sep_rate) if s]
            for agent in to_separate:
                firm = self.firms.get(agent.firm_id)
                if firm is not None and agent.id in firm.employees:
                    del firm.employees[agent.id]
                    firm.pending_replacements += 1
                agent.firm_id = None
                agent.set_commute(None)

        # Initiating Real Estate Market
        # Tax transaction taxes (ITBI) when selling house
        # Property tax (IPTU) collected. One twelfth per month
        house_prices = [h.price for h in self.houses.values()]
        house_price_quantiles = np.quantile(house_prices, q=np.cumsum(self.PARAMS["PERC_HOUSE_CATEGORIES"]).tolist())

        self.housing.housing_market(self, house_price_quantiles)
        for house in self.houses.values():
            house.pay_property_tax(self)

        # Family investments
        for fam in self.families.values():
            fam.invest(self.central, self.clock.year, self.clock.months, self.PARAMS)

        if self.PARAMS.get('BANK_NATIONAL', False):
            self.central.accrue_deposit_interest(datetime.date(self.clock.year, self.clock.months, 1))
            self.central.settle_with_national_bank()
        else:
            # Remunerate central bank idle liquid assets
            self.central.remunerate_liquid_balance()
        # Using all collected taxes to improve public services
        bank_taxes = self.central.collect_taxes()

        # Separate funds for region index update and separate for the policy case. Also, buy from intermediate market
        self.funds.invest_taxes(self.clock.year, bank_taxes)

        # Apply policies (when they are tested)
        self.funds.apply_policies()

        # Pass monthly information to be stored in Statistics
        self.output.save_stats_report(self, bank_taxes)
        self.stats.update_funds_base(self.clock.year)
        # Getting regional GDP
        self.output.save_regional_report(self)

        if conf.RUN["SAVE_DATA_PERIDIOCITY"] == "MONTHLY":
            self.output.save_data(self)

        if conf.RUN["PRINT_STATISTICS_AND_RESULTS_DURING_PROCESS"]:
            self.logger.info(self.clock.days)

    def quarterly(self):
        if conf.RUN["SAVE_DATA_PERIDIOCITY"] == "QUARTERLY":
            self.output.save_data(self)

    def yearly(self):
        if conf.RUN["SAVE_DATA_PERIDIOCITY"] == "ANNUALLY":
            self.output.save_data(self)


def compute_region_price_stats(houses):
    """
    Precompute approximate region-level housing prices.
    Returns:
        dict: region_id -> dict with summary stats
    """
    prices_per_size_by_region = defaultdict(list)

    for h in houses:
        prices_per_size_by_region[h.region_id].append(h.price / h.size)

    region_stats = {}
    for region_id, prices_per_size in prices_per_size_by_region.items():
        if prices_per_size:
            region_stats[region_id] = {
                "median": np.median(prices_per_size),
            }
        else:
            region_stats[region_id] = {
                "median": 0.0,
            }

    return region_stats