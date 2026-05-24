import copy
import datetime
from collections import defaultdict
from unicodedata import category

import numpy as np
import pandas as pd
from dateutil import relativedelta

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

emissions = pd.read_csv('input/emissions_sectors.csv', dtype={'mun_code': str})

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
            env_efficiency = 1
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
        self.input_inventory = copy.deepcopy(initial_input_sectors)
        # Amount monthly sold by the firm
        self.amount_sold = amount_sold
        self.product_index = product_index
        self.create_product()
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


    # These getters assume just one Product per firm
    @property
    def total_quantity(self):
        return self.inventory[0].quantity

    @property
    def prices(self):
        return self.inventory[0].price

    @total_quantity.setter
    def total_quantity(self, value):
        self.inventory[0].quantity = value

    @prices.setter
    def prices(self, value):
        return self.inventory[0].price

    # ECOLOGICAL PROCEDURES ###########################################################################################
    def probability_success(self, eco_investment, eco_lambda):
        """ 
        Returns the probability of success given the amount invested per revenue (I/R)
        """
        return 1 - np.exp(np.clip(- eco_lambda * eco_investment, -700, 700))

    def create_externalities(self, regions, tax_emission, emissions_param):
        """
        Based on empirical data, creates externalities according to money output produced by a given activity.
        Total emissions are multiplied by firm-level env efficiency.
        """
        # Environmental indicators (emissions, water, energy, waste) by municipality and sector
        # Using median from 2010.
        # Procedure: Apply endogenous salary amount to external ecoefficiency to find estimated output indicator
        if not self.no_emissions:
            emissions_this_month = self.env_efficiency * self.emissions_base * (self.revenue-self.input_cost) / emissions_param
            self.last_emissions = emissions_this_month
            self.env_indicators['emissions'] += emissions_this_month
            emission_tax = emissions_this_month * tax_emission
            if emission_tax > 0:
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
        eco_investment, paid_subsidies = self.decision_on_eco_efficiency(regional_market, regions)

        # Check if firm has enough balance
        eco_investment = max(0, min(self.total_balance, eco_investment))
        self.total_balance -= eco_investment

        # Stochastic process to actually reduce firm-level parameter
        params = regional_market.sim.PARAMS
        # Probability uses I/R (investment as fraction of revenue) as designed by the FOC
        investment_share = eco_investment / max(self.revenue, 1e-6)
        p_success = self.probability_success(investment_share, params['ECO_INVESTMENT_LAMBDA'])
        if p_success > seed_np.rand():
            self.env_efficiency *= params['ENVIRONMENTAL_EFFICIENCY_STEP']
        regions[self.region_id].collect_taxes(-paid_subsidies, "emissions")
        self.total_balance += paid_subsidies
        self.inno_inv = eco_investment

    def decision_on_eco_efficiency(self,regional_market,regions):
        """ 
        Choose how much to invest based on expected emission cost (taxes, reputational costs and intrinsic cost)
        Also accounts for possible environmental policies
        """
        params = regional_market.sim.PARAMS
        today = regional_market.sim.clock.days
        ## Calculate expected emission cost with adaptative expectations
        # Tax cost
        tax_cost = self.emission_taxes_paid
        input_cost = self.input_cost

        # TODO: Implement other emission costs
        reputation_cost, intrinsic_cost = 0,0
        total_cost = tax_cost+reputation_cost+intrinsic_cost+input_cost

        # The next step assumes linearity in costs
        # TODO: Define wether costs are linear or not: we can make any function over total_emission and have
        # expected_cost_reduction = cost(last_emissions)-cost((1-delta)*last_emissions)
        expected_cost_reduction = (1-params['ENVIRONMENTAL_EFFICIENCY_STEP']) * total_cost
        # Skip if within grace period
        is_policy_active = today > params['STARTING_DAY'] + datetime.timedelta(params['ECO_POLICY_DAYS'])
        eco_lambda = params['ECO_INVESTMENT_LAMBDA']
        subsidies = params['ECO_INVESTMENT_SUBSIDIES'][self.sector] if is_policy_active else 0
        if is_policy_active and subsidies and tax_cost >= 0:
            # Check if the government has money to provide subsidies
            # Only checks if emissions taxes are being levied
            if regions[self.region_id].treasure['emissions'] <= 0:
                subsidies = 0


        # Profit maximization FOC yields optimal absolute investment I = R·ln(λ·ECR/((1−s)·R))/λ
        if self.revenue > 0:
            eco_investment = (np.log(
                                eco_lambda * expected_cost_reduction / ((1 - subsidies) * self.revenue)) *
                              (self.revenue / eco_lambda))
        else:
            eco_investment = 0

        if eco_investment < 0:
            eco_investment = 0
        paid_subsidies = subsidies * eco_investment
        return eco_investment, paid_subsidies

    # PRODUCTION DEPARTMENT ###########################################################################################
    def choose_firm_per_sector(self, regional_market, firms, seed, market_size,
                               prebuilt_sector_map=None):
        """
        Choose local firms to buy inputs from, optimizing firm selection per sector.
        Accepts an optional prebuilt_sector_map (sector→list of firms) built once per month
        at the simulation level to avoid rebuilding it for every firm call.
        """
        chosen_firms = {}

        if prebuilt_sector_map is None:
            # Fallback: build the map here (slow path)
            sector_map = {}
            for firm in firms.values():
                if firm.id != self.id:
                    sector_map.setdefault(firm.sector, []).append(firm)
        else:
            sector_map = prebuilt_sector_map

        for sector in regional_market._sector_order:
            # Filter by positive inventory and exclude self; direct inventory[0].quantity
            # access avoids the property dispatch overhead on this hot path
            available_firms = [f for f in sector_map.get(sector, [])
                                if f.id != self.id and f.inventory[0].quantity > 0]

            if not available_firms:
                chosen_firms[sector] = None
                continue

            sampled_firms = seed.sample(available_firms, min(len(available_firms), int(market_size)))
            sampled_firms.sort(key=lambda f: f.inventory[0].price)
            chosen_firms[sector] = sampled_firms[:int(market_size)]

        return chosen_firms

    def buy_inputs(self, desired_quantity, regional_market, firms, seed,
                   technical_matrix, external_technical_matrix,
                   prebuilt_sector_map=None):

        self.input_cost = 0
        if self.total_balance <= 0:
            return

        params = regional_market.sim.PARAMS
        freight_cost = 1.0 + params['REGIONAL_FREIGHT_COST']
        sectors = regional_market._sector_order

        # Use pre-computed numpy column arrays (avoids pandas.loc on every call)
        local_tc = regional_market._tech_np[self.sector]       # shape (12,)
        external_tc = regional_market._ext_local_np[self.sector]
        total_tc = local_tc + external_tc
        input_ratio = np.where(total_tc > 0, local_tc / total_tc, 0.0)

        # Build inventory and quantity arrays without creating pd.Series
        inv_arr = np.array([self.input_inventory[s] for s in sectors])
        gross_needed = desired_quantity * total_tc
        net_needed_clipped = np.maximum(gross_needed - inv_arr, 0.0)
        local_needed = input_ratio * net_needed_clipped
        external_needed = net_needed_clipped - local_needed

        # Firm selection
        chosen_firms = self.choose_firm_per_sector(
            regional_market, firms, seed,
            params['INTERMEDIATE_SIZE_MARKET'],
            prebuilt_sector_map
        )

        # Compute total money needed
        money_local_inputs = 0.0
        money_external_inputs = 0.0

        for i, sector in enumerate(sectors):
            firms_sector = chosen_firms[sector]
            if not firms_sector:
                continue
            price = firms_sector[0].inventory[0].price
            money_local_inputs += local_needed[i] * price
            money_external_inputs += external_needed[i] * price * freight_cost

        total_money_needed = money_local_inputs + money_external_inputs

        if total_money_needed > 0:
            reduction_factor = min(self.total_balance, total_money_needed) / total_money_needed
        else:
            reduction_factor = 1.0

        self.total_balance -= reduction_factor * total_money_needed

        # Purchase loop
        for i, sector in enumerate(sectors):
            firms_sector = chosen_firms[sector]

            if firms_sector:
                price = firms_sector[0].inventory[0].price
            else:
                price = regional_market.sim.avg_prices

            money_local = reduction_factor * local_needed[i] * price
            money_external = reduction_factor * external_needed[i] * price * freight_cost

            if money_local == 0 and money_external == 0:
                continue

            # Local purchases
            if firms_sector:
                change = 0.0
                per_firm_money = money_local / len(firms_sector)
                for firm in firms_sector:
                    change += regional_market.intermediate_consumption(
                        per_firm_money, firm
                    )
            else:
                change = money_local

            # Freight adjustment
            freight_extra = (freight_cost - 1.0) * change
            if self.total_balance > freight_extra:
                self.total_balance -= freight_extra
                money_external += freight_cost * change
            else:
                money_external += self.total_balance
                self.total_balance = 0.0

            # External purchase
            regional_market.sim.external.intermediate_consumption(
                money_external, price * freight_cost
            )

            # Inventory + cost update
            self.input_inventory[sector] += (
                    (money_local - change) / price +
                    money_external / (price * freight_cost)
            )

            self.input_cost += money_local - change + money_external

    def update_product_quantity(self, prod_exponent, prod_divisor, regional_market, firms, seed,
                               prebuilt_sector_map=None):
        """
        Based on the MIP sector, buys inputs to produce a given money output of the activity, creates externalities
        and creates a price based on cost.
        """
        quantity = 0
        if self.employees and self.inventory:
            desired_quantity = self.total_qualification(prod_exponent) / prod_divisor

            technical_matrix = regional_market.technical_matrix
            external_technical_matrix = regional_market.ext_local_matrix

            # Buy inputs fills up input_inventory
            # Env efficiency reduces the amount of inputs needed, so the firms buys less
            self.buy_inputs(self.env_efficiency * desired_quantity, regional_market, firms, seed,
                            technical_matrix, external_technical_matrix, prebuilt_sector_map)

            # Leontief production constraint using pre-computed numpy arrays
            sectors = regional_market._sector_order
            local_tc = regional_market._tech_np[self.sector]
            external_tc = regional_market._ext_local_np[self.sector]
            input_quantities_needed = self.env_efficiency * desired_quantity * (local_tc + external_tc)

            inv_arr = np.array([self.input_inventory[s] for s in sectors])
            productive_constraint = np.where(
                input_quantities_needed > 0,
                np.clip(inv_arr / input_quantities_needed, 0.0, 1.0),
                1.0  # no input needed for this sector → no constraint
            )
            productive_constraint_numeric = max(float(productive_constraint.min()), 0.0)

            input_used = productive_constraint_numeric * input_quantities_needed
            for i, sector in enumerate(sectors):
                self.input_inventory[sector] -= input_used[i]

            quantity = productive_constraint_numeric * desired_quantity
            self.total_quantity += quantity
            self.amount_produced += quantity
        return quantity

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
            inventory_target_ratio=0.0,
            price_markup_cap=0.25,
    ):
        """ Update prices based on inventory and average prices
            Save signal for the labor market """
        # Sticky prices (KLENOW, MALIN, 2010)
        if seed_np.rand() < sticky_prices:
            for p in self.inventory.values():
                delta_price = seed_np.randint(0, int(2 * markup * 100) + 1) / 100
                productive_capacity = self.total_qualification(prod_exponent) / prod_magnitude_divisor
                # Firms target a safety-stock buffer above bare productive capacity.
                low_inventory = (self.total_quantity + productive_capacity) <= self.amount_sold * (1 + inventory_target_ratio)
                if low_inventory:
                    self.increase_production = True
                    # Rise freely up to avg_prices * (1 + cap); spatial monopoly premium bounded.
                    ceiling = avg_prices * (1 + price_markup_cap)
                    if p.price < ceiling:
                        p.price = min(p.price * (1 + delta_price), ceiling)
                else:
                    self.increase_production = False  # Lengnick
                    # Fall only if above average, damped by price_ruggedness.
                    if p.price > avg_prices:
                        p.price *= 1 - delta_price * price_ruggedness
        self.prices = sum(p.price for p in self.inventory.values()) / len(
            self.inventory
        )

    def reset_amount_sold(self):
        # Resetting amount sold to record monthly amounts
        self.amount_sold = 0
        self.revenue = 0

    def sale(self, amount, regions, tax_consumption, consumer_region_id, if_origin, external=False):
        """Sell max amount of products for a given amount of money.
        Each firm always carries exactly one product (inventory key 0), so the original
        loop over inventory is replaced with a direct access.
        """
        if amount > 0:
            product = self.inventory[0]
            if product.quantity > 0:
                bought_quantity = amount / product.price
                actual_amount = amount
                if bought_quantity > product.quantity:
                    bought_quantity = product.quantity
                    actual_amount = bought_quantity * product.price

                product.quantity -= bought_quantity
                revenue = actual_amount * (1 - tax_consumption)
                self.total_balance += revenue
                self.revenue += revenue

                if if_origin:
                    regions[self.region_id].collect_taxes(actual_amount * tax_consumption, "consumption")
                else:
                    if not external:
                        regions[consumer_region_id].collect_taxes(actual_amount * tax_consumption, "consumption")

                self.amount_sold += bought_quantity
                return amount - actual_amount  # change/refund to buyer
        # No stock or zero amount: full refund
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
        taxes = (self.revenue - self.wages_paid - self.input_cost) * tax_firm
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
        # guarantees that firms do not distribute all money when unemployment is 0
        # Calculating wage base on a per-employee basis.
        unemployment = .04 if unemployment == 0 else unemployment
        # Cold-start fallback: in months with zero sales revenue (typically month 1),
        # advance a small fraction of capital as implicit revenue so workers receive
        # non-zero wages and bootstrap household permanent income.
        effective_revenue = self.revenue if self.revenue > 0 else self.total_balance * 0.001
        # Exponential discount: labor_share = exp(-u * relevance). Approaches 0 asymptotically
        # as unemployment rises; equals ~0.94 at equilibrium 4% unemployment (same as old linear).
        # Replaces linear (1 - u*relevance) which crossed zero at u = 1/relevance ≈ 67%.
        labor_share = np.exp(-unemployment * relevance_unemployment)
        if self.num_employees > 0:
            return ((effective_revenue - self.input_cost) / self.num_employees) * labor_share
        else:
            return (effective_revenue - self.input_cost) * labor_share

    def make_payment(self, regions, unemployment, alpha, tax_labor, relevance_unemployment, tax_transport=False):
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
                    if tax_transport:
                        if self.num_employees > 10:
                            transport_tax = wage * tax_transport
                            wage -= transport_tax
                            regions[self.region_id].collect_taxes(transport_tax, "transport")
                    employee.money += wage
                    employee.last_wage = wage

                # Transfer collected LABOR TAXES to region
                labor_tax = total_salary_paid * tax_labor
                regions[self.region_id].collect_taxes(labor_tax, "labor")
                self.total_balance -= total_salary_paid
                self.wages_paid = total_salary_paid
            else:
                self.wages_paid = 0

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
        # Seed initial stock so the Leontief self-loop doesn't dead-lock at startup.
        # Without this, cities whose construction matrix is dominated by a local self-loop
        # (like Brasilia) never produce their first unit and never build any houses.
        self.total_quantity = 200

    # Realistic median floor areas (m²) by quality tier, matching Brazilian housing:
    # quality 1 ≈ MCMV Faixa 1 (~45 m²), quality 4 ≈ upscale (~160 m²).
    # Lognormal (mu, sigma) pairs; sigma=0.4 gives realistic spread without extreme outliers.
    _SIZE_PARAMS = {
        1: (np.log(45),  0.4),
        2: (np.log(70),  0.4),
        3: (np.log(110), 0.4),
        4: (np.log(160), 0.4),
    }

    def plan_house(self, regions, params, sim, seed_np, vacancy):
        """Decide where to build with which attributes"""
        # Probability depends on size of market
        # Construction responds more strongly to vacancy
        build_sensitivity = params['BUILD_VACANCY_SENSITIVITY']
        # Exponential suppression: probability of skipping grows smoothly with vacancy,
        # asymptotically approaching 1 but never reaching it — construction becomes very rare
        # at high vacancy without ever being categorically forbidden.
        # At 8% vacancy (equilibrium) and sensitivity=6: ~38% skip. At 25%: ~78% skip.
        if seed_np.rand() < 1.0 - np.exp(-vacancy * build_sensitivity):
            return

        # Check whether production capacity does not exceed hired construction
        # for the next construction cash flow period
        monthly_productivity_capacity = self.total_qualification(params["PRODUCTIVITY_EXPONENT"])
        if monthly_productivity_capacity == 0:
            self.increase_production = True

        if not self.building and not self.houses_for_sale:
            # Start building plan
            self.increase_production = True

        elif len(self.houses_for_sale) > params['MAX_HOUSE_STOCK']:
            return

        # Candidate regions for licenses and check of funds to buy license
        regions = [
            r for r in regions
            if r.licenses > 0
            and self.total_balance > r.license_price
        ]
        if not regions:
            return

        # Draw quality first, then size conditional on quality.
        # Low quality → small footprint (cheap absolute price, accessible to FGTS families).
        # High quality → large footprint (high absolute price, upscale market).
        b, c, d = 0.38, 0.3, 0.1
        building_quality = seed_np.choice([1, 2, 3, 4], p=[1 - (b + c + d), b, c, d])
        mu, sigma = self._SIZE_PARAMS[building_quality]
        building_size = seed_np.lognormal(mu, sigma)

        # Number of product quantities needed for the house
        gross_cost = building_size * building_quality
        # Productivity is drawn once per firm from [1 - MULTIPLIER*MARKUP, 1.0].
        # Lower productivity → higher building cost → fewer profitable projects.
        # Productivity reduces the cost of construction and sets the size of profiting when selling
        if not self.productivity:
            self.productivity = seed_np.randint(100 - int(params['CONSTRUCTION_FIRM_MARKUP_MULTIPLIER'] *
                                                params["MARKUP"] * 100), 101) / 100
        building_cost = gross_cost * self.productivity

        # Choose region where construction is most profitable.
        # Expected revenue uses quality-specific price per sqm (quality × region.index),
        # not the market median — cheap houses sell cheap, expensive houses sell dear.
        # Since license_price == region.index, profit simplifies to:
        #   size × quality × index × (1 − productivity × (1 + LOT_COST))
        # so margin is the same fraction for every quality tier; what differs is the
        # absolute project size (and therefore capital required and profit in units).
        profitable_regions = []
        for r in regions:
            expected_price = building_quality * r.index * building_size
            profit = expected_price - (
                    r.license_price * building_cost * (1 + params["LOT_COST"])
            )

            if profit > 0:
                profitable_regions.append(r)

        if not profitable_regions:
            return

        # Building in any profitable region
        region = sim.seed.choice(profitable_regions)
        idx = max(self.building) + 1 if self.building else 0
        self.building[idx]["region"] = region.id
        self.building[idx]["size"] = building_size
        self.building[idx]["quality"] = building_quality
        # Product.quantity increases as construction moves forward and is deducted at once.
        # Divided by HOUSE_PRODUCTION_ADEQUACY to bridge the scale gap between labour output
        # units (~3-8/month) and building_size in square metres (~60-200 m²).
        self.building[idx]["cost"] = building_cost * region.license_price / params["HOUSE_PRODUCTION_ADEQUACY"]

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
        paid = min(building_info["cost"], self.total_quantity)
        self.total_quantity -= paid

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
        self.revenue = self.cash_flow[self.present]
        # Using temporary planned income before money starts to flow in
        if self.revenue == 0 and self.monthly_planned_revenue:
            # Adding the last planned house income
            self.revenue = self.monthly_planned_revenue[-1]
        # Observing global economic performance has the added advantage of not spending all revenue on salaries
        if self.num_employees > 0:
            return (self.revenue / self.num_employees) * (
                    1 - (unemployment * relevance_unemployment)
            )
        else:
            return self.revenue * (1 - (unemployment * relevance_unemployment))

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
        self._transfer_prev = 0.0
        self._transfer_current = 0.0

    def reset_amount_sold(self):
        super().reset_amount_sold()
        self._transfer_prev = self._transfer_current
        self._transfer_current = 0.0

    def consume(self, sim):
        # As long as we provide labor and total_balance, the other methods are OK to use methods from regular firm
        # Consumption: government own consumption is used as update index. Other sectors consume here.
        total_consumption = defaultdict(float)

        execution_rate = sim.PARAMS.get('GOVERNMENT_EXECUTION_RATE', 1.0)
        money_to_spend = self.total_balance * execution_rate
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
            market = [firm for firm in market if firm.total_quantity > 0]
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

    def wage_base(self, unemployment, relevance_unemployment):
        # Wages are funded by last month's tax transfer, not total_balance.
        # total_balance conflates initial firm capitalization (large, from world generation)
        # with ongoing tax revenue, causing a massive month-1 wage spike if used directly.
        # invest_taxes() runs after make_payment() in the monthly cycle, so this month's
        # transfer is unavailable; _transfer_prev carries the previous month's amount.
        if self.num_employees == 0 or self._transfer_prev <= 0:
            return 0.0
        unemployment = .04 if unemployment == 0 else unemployment
        labor_share = np.exp(-unemployment * relevance_unemployment)
        return self._transfer_prev / self.num_employees * labor_share

    def assign_proportion(self, value):
        self.budget_proportion = value

    def government_transfer(self, amount):
        """ Government tax transfers credited directly as revenue (no inventory gate) """
        if amount > 0:
            self.total_balance += amount
            self.revenue += amount
            self._transfer_current += amount
        return 0
