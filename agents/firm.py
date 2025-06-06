import copy
import datetime
from collections import defaultdict
from unicodedata import category

import numpy as np
import pandas as pd
from dateutil import relativedelta
from numba import vectorize, float64


from .house import House
from .product import Product

np.seterr(divide='ignore', invalid='ignore')
initial_input_sectors = {'Agriculture': 0,
                         'Mining': 0,
                         'Manufacturing': 0,
                         'Utilities': 0,
                         'Construction': 0,
                         'Trade': 0,
                         'Transport': 0,
                         'Business': 0,
                         'Financial': 0,
                         'RealEstate': 0,
                         'OtherServices': 0,
                         'Government': 0
                         }

emissions = pd.read_csv('input/emissions_sector_average_years.csv')

@vectorize
def clip4(x, l, u):
    return max(min(x, u), l)

class Firm:
    """
    Firms contain all elements connected with firms, their methods to handle production, adding, paying
    and firing employees, maintaining information about their current staff, and available products, as
    well as cash flow. Decisions are based on endogenous variables and products are available when
    searched for by consumers.
    """

    def __init__(
            self,
            _id,
            address,
            total_balance,
            region_id,
            profit=1,
            amount_sold=0,
            product_index=0,
            amount_produced=0,
            total_quantity=0,
            wages_paid=0,
            present=datetime.date(2000, 1, 1),
            revenue=0,
            taxes_paid=0,
            input_cost=0,
            prices=None,
            sector=None,
            env_indicators=None,
            env_efficiency=1
    ):
        if env_indicators is None:
            self.env_indicators = {'emissions': 0}
        self.env_efficiency = env_efficiency
        self.increase_production = False
        self.id = _id
        self.address = address
        self.total_balance = total_balance
        self.region_id = region_id
        self.profit = profit
        self.input_cost = input_cost
        # Pool of workers in a given firm
        self.employees = {}
        # Firms makes existing products from class Products.
        # Products produced are stored by product_id in the inventory
        self.inventory = {}
        self.input_inventory, self.external_input_inventory = (copy.deepcopy(initial_input_sectors),
                                                               copy.deepcopy(initial_input_sectors))
        self.total_quantity = total_quantity
        # Amount monthly sold by the firm
        self.amount_sold = amount_sold
        self.product_index = product_index
        # Cumulative amount produced by the firm
        self.amount_produced = amount_produced
        self.wages_paid = wages_paid
        self.present = present
        # Monthly income received from sales
        self.revenue = revenue
        self.taxes_paid = taxes_paid
        self.emission_taxes_paid = 0
        self.prices = prices
        self.sector = sector
        self.no_emissions = False
        self.last_emissions = 0
        self.inno_inv = 0
        try:
            self.emissions_base = emissions[emissions.isic_12 == self.sector]['eco'].reset_index(drop=True)[0]
        except KeyError:
            self.no_emissions = True

    # Product procedures ##############################################################################################
    def create_product(self):
        """Check for and create new products.
        Products are only created if the firms' balance is positive."""
        if self.profit > 0:
            dummy_quantity = 0
            dummy_price = 1
            if self.product_index not in self.inventory:
                self.inventory[self.product_index] = Product(
                    self.product_index, dummy_quantity, dummy_price
                )
                self.product_index += 1
            self.prices = sum(p.price for p in self.inventory.values()) / len(self.inventory)

    # ECOLOGICAL PROCEDURES ###########################################################################################
    def probability_success(self, eco_investment, eco_lambda):
        """ 
        Returns the probability of success given the amount invested per wages paid (I/W)
        """
        return 1 - np.exp(- eco_lambda * eco_investment)

    def create_externalities(self, regions, tax_emission, emissions_param):
        """
        Based on empirical data, creates externalities according to money output produced by a given activity.
        Total emissions are multiplied by firm-level env efficiency.
        """
        # Environmental indicators (emissions, water, energy, waste) by sector
        # Procedure: Apply endogenous salary amount to external eco-efficiency to find estimated output indicator
        if not self.no_emissions:
            emissions_this_month = self.env_efficiency * emissions_param * self.wages_paid / self.emissions_base
            self.last_emissions = emissions_this_month
            self.env_indicators['emissions'] += emissions_this_month
            emission_tax = emissions_this_month * tax_emission
            if emission_tax >= 0:
                self.emission_taxes_paid = emission_tax
                self.total_balance -= emission_tax
                regions[self.region_id].collect_taxes(emission_tax, "emissions")
            else:
                self.emission_taxes_paid = 0

    def invest_eco_efficiency(self, regional_market, regions, seed_np):
        """
        Reduce overall emissions per wage employed.
        """
        # Decide how much to invest based on expected cost and benefit analysis
        eco_investment, paid_subsidies = self.decision_on_eco_efficiency(regional_market)
        
        # Check if firm has enough balance
        if self.total_balance < eco_investment * self.wages_paid or eco_investment==0:
            return  # No money to invest

        self.total_balance -= eco_investment * self.wages_paid - paid_subsidies
        regions[self.region_id].collect_taxes(-paid_subsidies, "emissions")
        self.inno_inv = eco_investment


        params = regional_market.sim.PARAMS
        # Stochastic process to actually reduce firm-level parameter
        p_success = self.probability_success(eco_investment, params['ECO_INVESTMENT_LAMBDA'])  # regional_market.
        random_value = seed_np.rand()
        if p_success > random_value:
            # Innovation was successful
            self.env_efficiency *= params['ENVIRONMENTAL_EFFICIENCY_STEP']
        else:
            # Nothing happens
            pass
        

    def decision_on_eco_efficiency(self, regional_market):
        """ 
        Choose how much to invest based on expected emission cost (taxes, reputational costs and intrinsic cost)
        Also accounts for possible environmental policies
        """
        params = regional_market.sim.PARAMS
        # Calculate expected emission cost with adaptively expectations
        # Tax cost
        tax_cost = self.emission_taxes_paid
        input_cost = self.input_cost

        total_cost = tax_cost + input_cost

        # The next step assumes linearity in costs
        expected_cost_reduction = (1 - params['ENVIRONMENTAL_EFFICIENCY_STEP']) * total_cost

        # Profit maximization formula yields the formula below
        eco_lambda, subsidies = params['ECO_INVESTMENT_LAMBDA'], params['ECO_INVESTMENT_SUBSIDIES']
        assert self.wages_paid >= 0
        inner_part_eco_investment = (eco_lambda * expected_cost_reduction) / ((1 - subsidies) * self.wages_paid) \
            if self.wages_paid > 0 else 0
        if inner_part_eco_investment > 1:
            investment_per_wages_paid = (np.log(inner_part_eco_investment) *
                                         (1 / eco_lambda))
        else:
            investment_per_wages_paid = 0
        # TODO: Can the government enter deficit?
        paid_subsidies = subsidies * investment_per_wages_paid * self.wages_paid
        return investment_per_wages_paid, paid_subsidies

    # PRODUCTION DEPARTMENT ###########################################################################################
    def choose_firm_per_sector(self, regional_market, firms, seed, market_size):
        """
        Choose local firms to buy inputs from
        """
        params = regional_market.sim.PARAMS
        chosen_firms = {}
        for sector in regional_market.technical_matrix.index:
            eligible_firms = [f for f in firms.values() if f.sector == sector and f.id != self.id]
            market = list(np.random.choice(eligible_firms, 
                              min(len(eligible_firms), int(params['SIZE_MARKET'])), 
                              replace=False))
            market = [firm for firm in market if firm.get_total_quantity() > 0]
            if market:
                # Choose firms with the cheapest average prices
                market.sort(key=lambda firm: firm.prices)
                # Choose the THREE cheapest firms, when available
                chosen_firms[sector] = market[:min(len(market), int(market_size))]
            else:
                chosen_firms[sector] = None
        return chosen_firms

    def buy_inputs(self, desired_quantity, regional_market, firms, seed,
                   technical_matrix, external_technical_matrix):
        """
        Buys inputs according to the technical coefficients.
        In fact, this is the intermediate consumer market (firms buying from firms)
        """
       
        # Reset input cost
        self.input_cost = 0
        if self.total_balance > 0:
            # First the firm checks how much it needs to buy
            params = regional_market.sim.PARAMS
            # The input ratio accounts for the need to buy inputs from other regions
            input_ratio = np.divide(technical_matrix.loc[:, self.sector],
                                    technical_matrix.loc[:, self.sector] +
                                    external_technical_matrix.loc[:, self.sector])
            input_quantities_needed = (desired_quantity *
                                       (technical_matrix.loc[:, self.sector] +
                                        external_technical_matrix.loc[:, self.sector]))
            input_quantities_needed -= pd.Series(self.input_inventory)
            input_quantities_needed = clip4(input_quantities_needed,0,10e6)#np.clip(input_quantities_needed, 0, None)
            local_input_quantities_needed = np.multiply(input_ratio,
                                                        input_quantities_needed)
            external_input_quantities_needed = input_quantities_needed - local_input_quantities_needed
            # Choose the firm to buy inputs from
            chosen_firms_per_sector = self.choose_firm_per_sector(regional_market, firms, seed,
                                                                  params['INTERMEDIATE_SIZE_MARKET'])
            money_local_inputs = sum([local_input_quantities_needed[sector] * chosen_firms_per_sector[sector][0].prices
                                      for sector in regional_market.technical_matrix.index
                                      if chosen_firms_per_sector[sector]])

            # External buying of inputs includes an ADDITIONAL FREIGHT COST!
            money_external_inputs = sum([external_input_quantities_needed[sector] *
                                         chosen_firms_per_sector[sector][0].prices *
                                         (1 + params['REGIONAL_FREIGHT_COST'])
                                         for sector in regional_market.technical_matrix.index
                                         if chosen_firms_per_sector[sector]])

            # The reduction factor is used to account for the firm having LESS MONEY than needed
            reduction_factor = (min(self.total_balance, money_local_inputs + money_external_inputs) /
                               (money_local_inputs + money_external_inputs)) if money_local_inputs + money_external_inputs > 0 else 1 


            # Withdraw all the necessary money. If no inputs are available, change is returned
            self.total_balance -= reduction_factor * (money_local_inputs + money_external_inputs)

            # First buy inputs locally. Pay cheapest firm prices
            for sector in regional_market.technical_matrix.index:
                if chosen_firms_per_sector[sector]:
                    prices = chosen_firms_per_sector[sector][0].prices
                else:
                    prices = regional_market.sim.avg_prices
                money_this_sector = (reduction_factor *
                                     local_input_quantities_needed[sector] *
                                     prices)

                # External money includes FREIGHT
                # TODO. Check flow consistency, where does FREIGHT MONEY GOES?
                external_money_this_sector = (reduction_factor * external_input_quantities_needed[sector] *
                                              prices *
                                              (1 + params['REGIONAL_FREIGHT_COST']))

                if money_this_sector == 0 and external_money_this_sector == 0:
                    continue
                # Uses regional market to access intermediate consumption and each firm sale function
                # Returns change, if any
                if chosen_firms_per_sector[sector]:
                    change = 0
                    # Buy inputs from all selected firms (from 1 to 3)
                    money_this_sector_this_firm = money_this_sector / len(chosen_firms_per_sector[sector])
                    for firm in chosen_firms_per_sector[sector]:
                        change += regional_market.intermediate_consumption(money_this_sector_this_firm,
                                                                           firm)

                else:
                    change = money_this_sector
                # Check whether there was change and buy the rest from the external sector
                #  so that firms won't consistently buy less than needed while having money
                if self.total_balance > ((1 + params['REGIONAL_FREIGHT_COST']) - 1) * change:
                    self.total_balance -= ((1 + params['REGIONAL_FREIGHT_COST']) - 1) * change
                    external_money_this_sector += (1 + params['REGIONAL_FREIGHT_COST']) * change
                else:
                    external_money_this_sector += self.total_balance
                    self.total_balance = 0
                # Go for external market which has full supply
                regional_market.sim.external.intermediate_consumption(external_money_this_sector,
                                                                      prices *
                                                                      (1 + params['REGIONAL_FREIGHT_COST']))
                self.input_inventory[sector] += ((money_this_sector - change) / prices +
                                                 external_money_this_sector / (prices *
                                                                               (1 + params['REGIONAL_FREIGHT_COST'])))
                self.input_cost += money_this_sector - change + external_money_this_sector

    def update_product_quantity(self, prod_exponent, prod_divisor, regional_market, firms, seed):
        """
        Based on the MIP sector, buys inputs to produce a given money output of the activity, creates externalities
        and creates a price based on cost.
        """
        # """ Production equation = Labor * qualification ** alpha """
        quantity = 0
        if self.employees and self.inventory:
            # Call get_sum_qualification below: sum([employee.qualification ** parameters.PRODUCTIVITY_EXPONENT
            #                                   for employee in self.employees.values()])
            # Divide production by an order of magnitude adjustment parameter
            desired_quantity = self.total_qualification(prod_exponent) / prod_divisor
            # Currently, each firm has only a single product. If more products should be introduced, allocation of
            # quantity per product should be adjusted accordingly
            # Currently, the index for the single product is 0

            technical_matrix = regional_market.technical_matrix
            external_technical_matrix = regional_market.ext_local_matrix

            # Buy inputs fills up input_inventory and external_input_inventory
            # Env efficiency reduces the amount of inputs needed, so the firms buys less
            self.buy_inputs(self.env_efficiency * desired_quantity, regional_market, firms, seed,
                            technical_matrix, external_technical_matrix)
            input_quantities_needed = self.env_efficiency * desired_quantity * (
                    technical_matrix.loc[:, self.sector] + external_technical_matrix.loc[:, self.sector])
            # The following process would be a traditional Leontief production function
            productive_constraint = np.divide(pd.Series(self.input_inventory),
                                              input_quantities_needed)
            productive_constraint = clip4(productive_constraint, 0, 1)#np.clip(productive_constraint, 0, 1)
            # Check that we have enough inputs to produce desired quantity
            productive_constraint_numeric = max(min(productive_constraint), 0)
            input_used = productive_constraint_numeric * input_quantities_needed
            quantity = productive_constraint_numeric * desired_quantity
            for sector in regional_market.technical_matrix.index:
                self.input_inventory[sector] -= input_used[sector]
            self.inventory[0].quantity += quantity
            self.amount_produced += quantity
        return quantity

    def get_total_quantity(self):
        # Simplifying for JUST ONE PRODUCT. More products will need rearranging it
        self.total_quantity = self.inventory[0].quantity
        return self.total_quantity

    # Commercial department
    def decision_on_prices_production(
            self,
            sticky_prices,
            markup,
            seed_np,
            avg_prices,
            prod_exponent=None,
            prod_magnitude_divisor=None,
            const_cash_flow=None,
            price_ruggedness=1,
    ):
        """ Update prices based on inventory and average prices
            Save signal for the labor market """
        # Sticky prices (KLENOW, MALIN, 2010)
        if seed_np.rand() > sticky_prices:
            for p in self.inventory.values():
                self.get_total_quantity()
                # if the firm has sold this month more than available in stocks, prices rise
                # Dawid 2018 p.26 Firm observes excess or shortage inventory and relative price considering other firms
                # Considering inventory to last one month only
                delta_price = seed_np.randint(0, int(2 * markup * 100) + 1) / 100
                productive_capacity = self.total_qualification(prod_exponent) / prod_magnitude_divisor
                low_inventory = (
                        ((self.total_quantity + productive_capacity) <= self.amount_sold) or self.total_quantity == 0
                )
                low_prices = p.price < avg_prices if avg_prices != 1 else True
                if low_inventory:
                    self.increase_production = True
                else:
                    self.increase_production = False  # Lengnick
                if low_inventory and low_prices:
                    p.price *= 1 + delta_price
                elif not low_inventory and not low_prices:
                    p.price *= 1 - delta_price * price_ruggedness
        self.prices = sum(p.price for p in self.inventory.values()) / len(
            self.inventory
        )

    def reset_amount_sold(self):
        # Resetting amount sold to record monthly amounts
        self.amount_sold = 0
        self.revenue = 0

    def sale(self, amount, regions, tax_consumption, consumer_region_id, if_origin, external=False):
        """Sell max amount of products for a given amount of money"""
        if amount > 0:
            # For each product in this firms' inventory, spend amount proportionally
            dummy_bought_quantity = 0
            amount_per_product = amount / len(self.inventory)

            # Add price of the unit, deduce it from consumers' amount
            for key in list(self.inventory.keys()):
                if self.inventory[key].quantity > 0:
                    bought_quantity = amount_per_product / self.inventory[key].price

                    # Verifying if demand is within firms' available inventory
                    if bought_quantity > self.inventory[key].quantity:
                        bought_quantity = self.inventory[key].quantity
                        amount_per_product = bought_quantity * self.inventory[key].price

                    # Deducing from stock
                    self.inventory[key].quantity -= bought_quantity

                    # Tax deducted from firms' balance and value of sale added to the firm
                    revenue = amount_per_product - (
                            amount_per_product * tax_consumption
                    )
                    self.total_balance += revenue
                    self.revenue += revenue

                    # Tax added to region-specific government.
                    # ATTENTION. this is the origin of consumption!
                    # For the new REFORM change it to the region of CONSUMER
                    if if_origin:
                        # Standard tax system. Consumption charged at firms' address
                        regions[self.region_id].collect_taxes(
                            amount_per_product * tax_consumption, "consumption"
                        )
                    else:
                        if external:
                            pass
                            # TODO: Add tax value to external region
                        else:
                            # Testing policy to charge consumption tax at consumers' address
                            regions[consumer_region_id].collect_taxes(
                                amount_per_product * tax_consumption, "consumption"
                            )
                    # Quantifying quantity sold
                    dummy_bought_quantity += bought_quantity
                    # Deducing money from clients upfront
                    amount -= amount_per_product
            self.amount_sold += dummy_bought_quantity
        # Return change to consumer, if any. Note that if there is no quantity to sell, full amount is returned
        return amount

    @property
    def num_products(self):
        return len(self.inventory)

    # Accountancy department ########################################################################################
    # Save values in time
    def calculate_profit(self):
        # Calculate profits considering last month wages paid and taxes on firm
        # (labor and consumption taxes are already deducted)
        self.profit = (self.revenue
                       - self.wages_paid
                       - self.taxes_paid
                       - self.input_cost
                       - self.emission_taxes_paid)

    def pay_taxes(self, regions, tax_firm):
        taxes = (self.revenue 
                    - self.wages_paid 
                    - self.input_cost 
                    - self.emission_taxes_paid) * tax_firm
        if taxes >= 0:
            # Revenue minus salaries paid in previous month may be negative.
            # In this case, no taxes are paid
            self.taxes_paid = taxes
            self.total_balance -= taxes
            regions[self.region_id].collect_taxes(self.taxes_paid, "firm")
        else:
            self.taxes_paid = 0

    # Employees' procedures #########
    def total_qualification(self, alpha):
        return sum(
            [employee.qualification ** alpha for employee in self.employees.values()]
        )

    def wage_base(self, unemployment, relevance_unemployment):
        # Observing global economic performance to set salaries,
        # guarantees that firms do not spend all revenue on salaries
        # Calculating wage base on a per-employee basis.
        if self.num_employees > 0:
            return ((self.revenue 
                     - self.input_cost
                     - self.emission_taxes_paid) / self.num_employees) * (max(
                    1 - (unemployment * relevance_unemployment), 0)
            )
        else:
            return (self.revenue - self.input_cost- self.emission_taxes_paid) * (max(
                    1 - (unemployment * relevance_unemployment), 0)
            )

    def make_payment(self, regions, unemployment, alpha, tax_labor, relevance_unemployment):
        """ Pay employees based on revenue, relative employee qualification, labor taxes, and alpha param
        """
        if self.employees:
            # Total salary, including labor taxes. Reinstating total salary paid by multiplying wage * num_employees
            total_salary_paid = (
                    self.wage_base(unemployment, relevance_unemployment)
                    * self.num_employees
            )
            if total_salary_paid > 0:
                total_qualification = self.total_qualification(alpha)
                for employee in self.employees.values():
                    # Making payment according to employees' qualification.
                    # Deducing it from firms' balance
                    # Deduce LABOR TAXES from employees' salaries as a percentual of each salary
                    wage = (
                                   total_salary_paid
                                   * (employee.qualification ** alpha)
                                   / total_qualification
                           ) * (1 - tax_labor)
                    employee.money += wage
                    employee.last_wage = wage

                # Transfer collected LABOR TAXES to region
                labor_tax = total_salary_paid * tax_labor
                regions[self.region_id].collect_taxes(labor_tax, "labor")
                self.total_balance -= total_salary_paid
                self.wages_paid = total_salary_paid * (1-tax_labor)
            else:
                self.wages_paid = 0
                for employee in self.employees.values():
                    employee.last_wage = 0

    # Human resources department #################
    def add_employee(self, employee):
        # Adds a new employee to firms' set
        # Employees are instances of Agents
        self.employees[employee.id] = employee
        employee.firm_id = self.id

    def obit(self, employee):
        del self.employees[employee.id]

    def fire(self, seed):
        if self.employees:
            employee = seed.choice(list(self.employees.values()))
            self.employees[employee.id].firm_id = None
            self.employees[employee.id].set_commute(None)
            del self.employees[employee.id]

    def is_worker(self, id_):
        # Returns true if agent is a member of this firm
        return id_ in self.employees

    @property
    def num_employees(self):
        return len(self.employees)

    def get_total_balance(self):
        return self.total_balance

    def __repr__(self):
        return "FirmID: %s, $ %d, Emp. %d, Quant. %d, Address: %s at %s" % (
            self.id,
            self.total_balance,
            self.num_employees,
            self.total_quantity,
            self.address,
            self.region_id,
        )


class AgricultureFirm(Firm):
    pass


class MiningFirm(Firm):
    pass


class ManufacturingFirm(Firm):
    pass


class UtilitiesFirm(Firm):
    pass


class ConstructionFirm(Firm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.houses_built = []
        self.houses_for_sale = []
        self.building = defaultdict(dict)
        self.cash_flow = defaultdict(float)
        self.monthly_planned_revenue = list()
        self.productivity = 0

    def plan_house(self, regions, houses, params, sim, seed_np, vacancy):
        """Decide where to build with which attributes"""
        # Probability depends on size of market
        if vacancy:
            if seed_np.rand() < vacancy:
                return

        # Check whether production capacity does not exceed hired construction
        # for the next construction cash flow period
        monthly_productivity_capacity = self.total_qualification(params["PRODUCTIVITY_EXPONENT"])
        if monthly_productivity_capacity == 0:
            self.increase_production = True
        if not self.building and not self.houses_for_sale:
            # Start building plan
            self.increase_production = True
            pass
        elif len(self.houses_for_sale) <= params['MAX_HOUSE_STOCK']:
            pass
        else:
            return

        # Candidate regions for licenses and check of funds to buy license
        regions = [
            r
            for r in regions
            if r.licenses > 0 and self.total_balance > r.license_price
        ]
        if not regions:
            return
        # Targets
        building_size = seed_np.lognormal(4.96, 0.5)
        b, c, d = 0.38, 0.3, 0.1
        building_quality = seed_np.choice([1, 2, 3, 4], p=[1 - (b + c + d), b, c, d])

        # Get information about regions' house prices
        region_ids = [r.id for r in regions]
        region_prices = defaultdict(list)
        for region_id in region_ids:
            for h in houses:
                # In correct region
                # within 100 size units,
                # within 2 quality
                if (
                        h.region_id in region_id
                        and abs(h.size - building_size) <= 100
                        and abs(h.quality - building_quality) < 2
                ):
                    region_prices[h.region_id].append(h.price)
                    # Only take a sample
                    if len(region_prices[region_id]) > 100:
                        break
            if len(region_prices[region_id]) == 0:
                region_prices.pop(region_id)

        # Number of product quantities needed for the house
        gross_cost = building_size * building_quality
        # Productivity of the company may vary double than exogenous set markup.
        # Productivity reduces the cost of construction and sets the size of profiting when selling
        if not self.productivity:
            self.productivity = seed_np.randint(100 - int(params['CONSTRUCTION_FIRM_MARKUP_MULTIPLIER'] *
                                                          params["MARKUP"] * 100), 101) / 100
        building_cost = gross_cost * self.productivity

        # Choose region where construction is most profitable
        # There might not be samples for all regions, so fallback to price of 0
        region_mean_prices = {
            r_id: sum(vs) / len(vs) for r_id, vs in region_prices.items()
        }
        # Using median prices for regions without price information
        if not region_mean_prices.values():
            median_prices = 0
        else:
            median_prices = np.median(list(region_mean_prices.values()))
        region_profitability = [
            region_mean_prices.get(r.id, median_prices)
            - (r.license_price * building_cost * (1 + params["LOT_COST"]))
            for r in regions
        ]
        regions = [(r, p) for r, p in zip(regions, region_profitability) if p > 0]

        # No profitable regions
        if not regions:
            return

        # Building in any profitable region
        region = sim.seed.choice([r[0] for r in regions])
        idx = max(self.building) + 1 if self.building else 0
        self.building[idx]["region"] = region.id
        self.building[idx]["size"] = building_size
        self.building[idx]["quality"] = building_quality

        # Product.quantity increases as construction moves forward and is deducted at once
        self.building[idx]["cost"] = building_cost * region.license_price

        # Provide temporary cashflow revenue numbers before sales start to trickle in.
        # Additional value per month. Expectations of monthly payments before first sell
        self.monthly_planned_revenue.append(
            self.building[idx]["cost"] / params["CONSTRUCTION_ACC_CASH_FLOW"]
        )

        # Buy license
        region.licenses -= 1
        # Region license price is current QLI. Lot price is the model parameter
        cost_of_land = region.license_price * building_cost * params["LOT_COST"]
        self.total_balance -= cost_of_land
        region.collect_taxes(cost_of_land, "transaction")

    def build_house(self, regions, generator):
        """Firm decides if house is finished"""
        if not self.building:
            return

        # Not finished
        min_cost_idx = [
            k for k in self.building if self.building[k]["cost"] < self.total_quantity
        ]
        if not min_cost_idx:
            return
        else:
            # Choose the house that entered earlier and for which there is enough material to instantaneously build.
            min_cost_idx = min(min_cost_idx)

        # Finished, expend inputs
        # Remember: if inventory of products is expanded for more than 1, this needs adapting
        building_info = self.building[min_cost_idx]
        paid = min(building_info["cost"], self.inventory[0].quantity)
        self.inventory[0].quantity -= paid

        # Choose random place in region
        region = regions[building_info["region"]]
        probability_urban = generator.seed_np.choice(
            [True, False],
            p=[generator.prob_urban(region), (1 - generator.prob_urban(region))],
        )
        if probability_urban:
            address = generator.get_random_points_in_polygon(
                generator.urban[region.id[:7]]
            )[0]
        else:
            address = generator.get_random_points_in_polygon(region)[0]
        # Create the house
        house_id = generator.gen_id()
        size = building_info["size"]
        quality = building_info["quality"]
        price = (size * quality) * region.index
        h = House(
            house_id,
            address,
            size,
            price,
            region.id,
            quality,
            owner_id=self.id,
            owner_type=House.Owner.FIRM,
        )

        # Archive the register of the completed house
        del self.building[min_cost_idx]

        # Register accomplishments within firms' house inventory
        self.houses_built.append(h)
        self.houses_for_sale.append(h)

        return h

    # Selling house
    def update_balance(self, amount, acc_months=None, date=datetime.date(2000, 1, 1)):
        self.total_balance += amount
        self.amount_sold += amount
        if acc_months is not None:
            acc_months = int(acc_months)
            for i in range(acc_months):
                self.cash_flow[date] += amount / acc_months
                date += relativedelta.relativedelta(months=+1)

    def wage_base(self, unemployment, relevance_unemployment):
        #TODO: All revenue added before (from intermediate and final consumption) is being ovelooked
        self.revenue += self.cash_flow[self.present]
        # Using temporary planned income before money starts to flow in
        if self.cash_flow[self.present] == 0 and self.monthly_planned_revenue:
            # Adding the last planned house income
            self.revenue += self.monthly_planned_revenue[-1]
        # Observing global economic performance has the added advantage of not spending all revenue on salaries
        if self.num_employees > 0:
            return ((self.revenue 
                     - self.input_cost
                     - self.emission_taxes_paid) / self.num_employees)  * max(
                    1 - (unemployment * relevance_unemployment), 0)
        else:
            return (self.revenue-self.input_cost-self.emission_taxes_paid) * (1 - (unemployment * relevance_unemployment))

    @property
    def n_houses_sold(self):
        return len(self.houses_built) - len(self.houses_for_sale)

    def mean_house_price(self):
        if not self.houses_built:
            return 0
        t = sum(h.price for h in self.houses_built)
        return t / len(self.houses_built)


class TradeFirm(Firm):
    pass


class TransportFirm(Firm):
    pass


class BusinessFirm(Firm):
    pass


class FinancialFirm(Firm):
    pass


class RealEstateFirm(Firm):
    pass


class OtherServicesFirm(Firm):
    pass


class GovernmentFirm(Firm):
    # Include special method for hiring/firing = fixed number
    # Include special method for setting prices, wages paying, profits, consume (supply total_balance)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget_proportion = 0

    def consume(self, sim):
        # As long as we provide labor and total_balance, the other methods are OK to use methods from regular firm
        # Consumption: government own consumption is used as update index. Other sectors consume here.
        total_consumption = defaultdict(float)

        money_to_spend = self.total_balance/10
        self.total_balance -= money_to_spend
        for sector in sim.regional_market.final_demand.index:
            if sector == 'Government':
                self.total_balance += money_to_spend * sim.regional_market.final_demand['GovernmentConsumption'][sector]
                # Government on consumption is operated as update_index at funds.py
                continue
            # This makes sure that only the final demand percentage of total balance is consumed at other sectors
            # Same as how households consume
            money_this_sector = money_to_spend * sim.regional_market.final_demand['GovernmentConsumption'][sector]
            # Some sectors have 0 value, such as Government, Mining, and Construction (explicit markets are used)
            if money_this_sector == 0:
                continue
            sector_firms = [f for f in sim.firms.values() if f.sector == sector]
            market = sim.seed.sample(sector_firms,
                                     min(len(sector_firms), int(sim.PARAMS['SIZE_MARKET'])))
            market = [firm for firm in market if firm.get_total_quantity() > 0]
            if market:
                chosen_firm = min(market, key=lambda firm: firm.prices)
                # Buy from chosen company
                change = chosen_firm.sale(money_this_sector, sim.regions, sim.PARAMS['TAX_CONSUMPTION'],
                                          self.region_id, sim.PARAMS["TAX_ON_ORIGIN"])
                self.total_balance += change
                total_consumption[sector] += money_this_sector - change
            else:
                self.total_balance += money_this_sector.copy()
        return total_consumption

    def assign_proportion(self, value):
        self.budget_proportion = value

    def government_transfer(self, amount):
        """ Equivalent to sales for regular firms,
            in which government transfer are added to government firms total balance """
        if amount > 0:
            # Add price of the unit, deduce it from consumers' amount
            for key in list(self.inventory.keys()):
                if self.inventory[key].quantity > 0:
                    amount_sold = self.inventory[key].quantity
                    # Deducing from stock
                    self.inventory[key].quantity = 0
                    self.total_balance += amount
                    self.revenue += amount
                    self.amount_sold += amount_sold
        return 0
