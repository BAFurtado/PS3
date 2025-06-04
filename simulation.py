import datetime
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
from world.firms import firm_growth
from world.funds import Funds
from world.geography import Geography, STATES_CODES, state_string
from markets.goods import RegionalMarket, External


class Simulation:
    def __init__(self, params, output_path):
        self.PARAMS = params
        self.geo = Geography(params, self.PARAMS["STARTING_DAY"].year)
        self.regional_market = RegionalMarket(self)
        self.clock = clock.Clock(self.PARAMS["STARTING_DAY"])
        self.output = analysis.Output(self, output_path)
        self.stats = analysis.Statistics(params)
        self.logger = analysis.Logger(hex(id(self))[-5:])
        self.funds = Funds(self)
        self._seed = (
            secrets.randbelow(2 ** 32)
            if conf.RUN["KEEP_RANDOM_SEED"]
            else conf.RUN.get("SEED", 0)
        )
        self.seed = random.Random(self._seed)
        self.seed_np = np.random.RandomState(self._seed)
        self.generator = Generator(self)
        # Generate the external supplier
        self.avg_prices = 1
        self.external = External(self, self.PARAMS["TAXES_STRUCTURE"]["consumption_equal"])
        self.mun_pops = dict()
        self.reg_pops = dict()
        self.grave = list()
        self.mun_to_regions = defaultdict(set)
        # Read necessary files
        self.m_men, self.m_women, self.f = dict(), dict(), dict()

        for state in self.geo.states_on_process:
            self.m_men[state] = pd.read_csv(
                "input/Demografia/2_Mortality/mortality_men_%s.csv" % state,
                header=0,
                decimal=".",
            ).groupby("age")
            self.m_women[state] = pd.read_csv(
                "input/Demografia/2_Mortality/mortality_women_%s.csv" % state,
                header=0,
                decimal=".",
            ).groupby("age")
            self.f[state] = pd.read_csv(
                "input/Demografia/1_Fertility/fertility_%s.csv" % state,
                header=0,
                decimal=".",
            ).groupby("age")
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

        # Count populations for each municipality and region
        for agent in agents.values():
            r_id = agent.region_id
            mun_code = r_id[:7]
            if r_id not in self.reg_pops:
                self.reg_pops[r_id] = 0
            if mun_code not in self.mun_pops:
                self.mun_pops[mun_code] = 0
            self.update_pop(None, r_id)
        return regions, agents, houses, families, firms, self.generator.central

    def run(self):
        """Runs the simulation"""
        self.logger.logger.info("Starting run.")
        self.logger.logger.info("Output: {}".format(self.output.path))
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

        # Group regions into their municipalities
        for region_id in self.regions.keys():
            mun_code = region_id[:7]
            self.mun_to_regions[mun_code].add(region_id)
        for mun_code, regions in self.mun_to_regions.items():
            self.mun_to_regions[mun_code] = list(regions)

        # Beginning of simulation, generate a product
        for firm in self.firms.values():
            firm.create_product()

        # First jobs allocated
        # Create an existing job market
        self.labor_market.look_for_jobs(self.agents)
        total = actual = self.labor_market.num_candidates
        actual_unemployment = self.stats.global_unemployment_rate
        # Simple average of 6 Metropolitan regions Brazil January 2000
        while actual / total > 0.086:
            self.labor_market.hire_fire(self.firms, 1, initialize=True)
            self.labor_market.assign_post(actual_unemployment, None, self.PARAMS)
            self.labor_market.look_for_jobs(self.agents)
            actual = self.labor_market.num_candidates
        self.labor_market.reset()

        # Update initial pop
        for region in self.regions.values():
            region.pop = self.reg_pops[region.id]

    def daily(self):
        pass

    def monthly(self):
        # Set interest rates
        values = self.interest[
            self.interest.index.date == self.clock.days][['interest', 'mortgage', 'sbpe', 'fgts']].iloc[0]
        self.central.set_interest(*values)

        current_unemployment = self.stats.global_unemployment_rate

        # Create new land licenses
        licenses_per_region = self.PARAMS["EXPECTED_LICENSES_PER_REGION"]
        for region in self.regions.values():
            region.licenses += self.seed_np.poisson(lam=licenses_per_region)

        # Create new firms according to average historical growth
        firm_growth(self)

        # Update firm products
        prod_exponent = self.PARAMS["PRODUCTIVITY_EXPONENT"]
        prod_magnitude_divisor = self.PARAMS["PRODUCTIVITY_MAGNITUDE_DIVISOR"]
        [f.reset_amount_sold() for f in self.firms.values()]
        for firm in self.firms.values():
            firm.update_product_quantity(prod_exponent, prod_magnitude_divisor,
                                         self.regional_market,
                                         self.firms,
                                         self.seed)

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
            firm.present = self.clock

        # FAMILIES CONSUMPTION -- using payment received from previous month
        # Equalize money within family members
        # Tax consumption when doing sales are realized
        self.regional_market.consume()
        # Government firms consumption
        self.regional_market.government_consumption()
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
        tax_emission = self.PARAMS["TAX_EMISSION"]
        relevance_unemployment = self.PARAMS["RELEVANCE_UNEMPLOYMENT_SALARIES"]
        sticky = self.PARAMS["STICKY_PRICES"]
        markup = self.PARAMS["MARKUP"]
        const_cash_flow = self.PARAMS["CONSTRUCTION_ACC_CASH_FLOW"]
        price_ruggedness = self.PARAMS["PRICE_RUGGEDNESS"]
        self.avg_prices, _ = self.stats.update_price(self.firms, mid_simulation_calculus=True)
        for firm in self.firms.values():
            # Tax workers when paying salaries
            firm.make_payment(
                self.regions,
                current_unemployment,
                prod_exponent,
                tax_labor,
                relevance_unemployment)
            # Firms update generated externalities, based on own sector and wages paid this month
            firm.create_externalities(self.regions, tax_emission, self.PARAMS['EMISSIONS_PARAM'])
            # Tax firms before profits: (revenue - salaries paid)
            firm.pay_taxes(self.regions, tax_firm)
            # Profits are after taxes
            firm.calculate_profit()
            # Check whether it is necessary to update prices
            firm.decision_on_prices_production(
                sticky,
                markup,
                self.seed_np,
                self.avg_prices,
                prod_exponent,
                prod_magnitude_divisor,
                const_cash_flow,
                price_ruggedness,
            )
            firm.invest_eco_efficiency(
                self.regional_market,
                self.regions,
                self.seed_np)

        # Construction firms
        # Probability depends (strongly) on market supply
        if self.PARAMS["OFFER_SIZE_ON_PRICE"]:
            vacancy = self.stats.vacancy_rate
        construction_firms = [f for f in self.firms.values() if f.sector == 'Construction']
        for firm in construction_firms:
            # See if firm can build a house
            firm.plan_house(
                self.regions.values(),
                self.houses.values(),
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
        self.labor_market.look_for_jobs(self.agents)

        # FIRMS
        # Government labor first (initialization is for all firms, government specific is monthly)
        self.labor_market.gov_hire_fire(self)
        # Check if new employee needed. Check if firing is necessary
        # 3-way criteria: Wages/sales, profits, and increase production
        self.labor_market.hire_fire(self.firms, self.PARAMS["LABOR_MARKET"])

        # Job Matching
        # Sample used only to calculate wage deciles
        sample_size = math.floor(len(self.agents) * 0.5)
        last_wages = [
            a.last_wage
            for a in list(self.seed.sample(list(self.agents.values()), sample_size))
            if a.last_wage is not None
        ]
        wage_deciles = np.percentile(last_wages, np.arange(10, 101, 10))
        self.labor_market.assign_post(current_unemployment, wage_deciles, self.PARAMS)

        # Initiating Real Estate Market
        self.logger.logger.info(
            f"Available licenses: {sum([r.licenses for r in self.regions.values()]):,.0f}"
        )
        # Tax transaction taxes (ITBI) when selling house
        # Property tax (IPTU) collected. One twelfth per month
        # self.central.calculate_monthly_mortgage_rate()
        house_price_quantiles = np.quantile(
            [h.price for h in self.houses.values()], q=[0.25, 0.5, 0.75]
        )

        self.housing.housing_market(self, house_price_quantiles)
        # (changed location) self.housing.process_monthly_rent(self)
        for house in self.houses.values():
            house.pay_property_tax(self)

        # Family investments
        for fam in self.families.values():
            fam.invest(self.central, self.clock.year, self.clock.months)

        # Using all collected taxes to improve public services
        bank_taxes = self.central.collect_taxes()

        # Separate funds for region index update and separate for the policy case. Also, buy from intermediate market
        self.funds.invest_taxes(self.clock.year, bank_taxes)

        # Apply policies if percentage is different from 0
        if self.PARAMS["POLICY_COEFFICIENT"]:
            self.funds.apply_policies()

        # Pass monthly information to be stored in Statistics
        self.output.save_stats_report(self, bank_taxes)

        # Getting regional GDP
        self.output.save_regional_report(self)

        if conf.RUN["SAVE_AGENTS_DATA"] == "MONTHLY":
            self.output.save_data(self)

        if conf.RUN["PRINT_STATISTICS_AND_RESULTS_DURING_PROCESS"]:
            self.logger.info(self.clock.days)

    def quarterly(self):
        if conf.RUN["SAVE_AGENTS_DATA"] == "QUARTERLY":
            self.output.save_data(self)

    def yearly(self):
        if conf.RUN["SAVE_AGENTS_DATA"] == "ANNUALLY":
            self.output.save_data(self)
