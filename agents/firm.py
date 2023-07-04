import datetime
from dateutil import relativedelta
from .house import House
from .product import Product
from collections import defaultdict


class Firm:
    """
    Firms contain all elements connected with firms, their methods to handle production, adding, paying
    and firing employees, maintaining information about their current staff, and available products, as
    well as cash flow. Decisions are based on endogenous variables and products are available when
    searched for by consumers.
    """
    type = 'CONSUMER'

    def __init__(self, _id,
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
                 prices=None):

        self.increase_production = False
        self.id = _id
        self.address = address
        self.total_balance = total_balance
        self.region_id = region_id
        self.profit = profit
        # Pool of workers in a given firm
        self.employees = {}
        # Firms makes existing products from class Products.
        # Products produced are stored by product_id in the inventory
        self.inventory = {}
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
        self.prices = prices

    # Product procedures ##############################################################################################
    def create_product(self):
        """Check for and create new products.
        Products are only created if the firms' balance is positive."""
        if self.profit > 0:
            dummy_quantity = 0
            dummy_price = 1
            if self.product_index not in self.inventory:
                self.inventory[self.product_index] = Product(self.product_index, dummy_quantity, dummy_price)
                self.product_index += 1
            self.prices = sum(p.price for p in self.inventory.values()) / len(self.inventory)

    # Production department
    def update_product_quantity(self, prod_expoent, prod_divisor):
        """Production equation = Labor * qualification ** alpha"""
        if self.employees and self.inventory:
            # Call get_sum_qualification below: sum([employee.qualification ** parameters.PRODUCTIVITY_EXPONENT
            #                                   for employee in self.employees.values()])

            # Divide production by an order of magnitude adjustment parameter
            quantity = self.total_qualification(prod_expoent) / prod_divisor
            # Currently, each firm has only a single product. If more products should be introduced, allocation of
            # quantity per product should be adjusted accordingly
            # Currently, the index for the single product is 0
            self.inventory[0].quantity += quantity
            self.amount_produced += quantity

    def get_total_quantity(self):
        self.total_quantity = sum(p.quantity for p in self.inventory.values())
        return self.total_quantity

    # Commercial department
    def decision_on_prices_production(self, sticky_prices, markup, seed, avg_prices,
                                      prod_exponent=None, prod_magnitude_divisor=None, const_cash_flow=None,
                                      price_ruggedness=1):
        """ Update prices based on inventory and average prices
            Save signal for the labor market
        """
        # Sticky prices (KLENOW, MALIN, 2010)
        if seed.random() > sticky_prices:
            for p in self.inventory.values():
                self.get_total_quantity()
                # if the firm has sold this month more than available in stocks, prices rise
                # Dawid 2018 p.26 Firm observes excess or shortage inventory and relative price considering other firms
                # Considering inventory to last one month only
                delta_price = (seed.randint(0, int(2 * markup * 100)) / 100)
                low_inventory = self.total_quantity <= self.amount_sold or self.total_quantity == 0
                low_prices = p.price < avg_prices if avg_prices != 1 else True
                if low_inventory:
                    self.increase_production = True
                else:
                    self.increase_production = False  # Lengnick
                if low_inventory and low_prices:
                    p.price *= (1 + delta_price)
                elif not low_inventory and not low_prices:
                    p.price *= 1 - delta_price * price_ruggedness
        # Resetting amount sold to record monthly amounts
        self.amount_sold = 0
        self.prices = sum(p.price for p in self.inventory.values()) / len(self.inventory)

    def sale(self, amount, regions, tax_consumption):
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
                    revenue = (amount_per_product - (amount_per_product * tax_consumption))
                    self.total_balance += revenue
                    self.revenue += revenue

                    # Tax added to region-specific government
                    regions[self.region_id].collect_taxes(amount_per_product * tax_consumption, 'consumption')

                    # Quantifying quantity sold
                    dummy_bought_quantity += bought_quantity

                    # Deducing money from clients upfront
                    amount -= amount_per_product
            self.amount_sold += dummy_bought_quantity
        # Return change to consumer, if any
        return amount

    @property
    def num_products(self):
        return len(self.inventory)

    # Accountancy department ########################################################################################
    # Save values in time
    def calculate_profit(self):
        # Calculate profits considering last month wages paid and taxes on firm
        # (labor and consumption taxes are already deducted)
        self.profit = self.revenue - self.wages_paid - self.taxes_paid

    def pay_taxes(self, regions, tax_firm):
        taxes = (self.revenue - self.wages_paid) * tax_firm
        if taxes >= 0:
            # Revenue minus salaries paid in previous month may be negative.
            # In this case, no taxes are paid
            self.taxes_paid = taxes
            self.total_balance -= taxes
            regions[self.region_id].collect_taxes(self.taxes_paid, 'firm')
        else:
            self.taxes_paid = 0

    # Employees' procedures #########
    def total_qualification(self, alpha):
        return sum([employee.qualification ** alpha for employee in self.employees.values()])

    def wage_base(self, unemployment, relevance_unemployment):
        # Observing global economic performance to set salaries
        # guarantees firms do not spend all revenue on salaries
        # Calculating wage base on a per-employee basis.
        if self.num_employees > 0:
            return (self.revenue / self.num_employees) * (1 - (unemployment * relevance_unemployment))
        else:
            return self.revenue * (1 - (unemployment * relevance_unemployment))

    def make_payment(self, regions, unemployment, alpha, tax_labor, relevance_unemployment):
        """Pay employees based on revenue, relative employee qualification, labor taxes, and alpha param"""
        if self.employees:
            # Total salary, including labor taxes. Reinstating total salary paid by multiplying wage * num_employees
            total_salary_paid = self.wage_base(unemployment, relevance_unemployment) * self.num_employees
            if total_salary_paid > 0:
                total_qualification = self.total_qualification(alpha)
                for employee in self.employees.values():
                    # Making payment according to employees' qualification.
                    # Deducing it from firms' balance
                    # Deduce LABOR TAXES from employees' salaries as a percentual of each salary
                    wage = (total_salary_paid * (employee.qualification ** alpha)
                            / total_qualification) * (1 - tax_labor)
                    employee.money += wage
                    employee.last_wage = wage

                # Transfer collected LABOR TAXES to region
                labor_tax = total_salary_paid * tax_labor
                regions[self.region_id].collect_taxes(labor_tax, 'labor')
                self.total_balance -= total_salary_paid
                self.wages_paid = total_salary_paid

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
            id = seed.choice(list(self.employees.keys()))
            self.employees[id].firm_id = None
            self.employees[id].set_commute(None)
            del self.employees[id]

    def is_worker(self, id):
        # Returns true if agent is a member of this firm
        return id in self.employees

    @property
    def num_employees(self):
        return len(self.employees)

    def get_total_balance(self):
        return self.total_balance

    def __repr__(self):
        return 'FirmID: %s, $ %d, Emp. %d, Quant. %d, Address: %s at %s' % (self.id,
                                                                            self.total_balance,
                                                                            self.num_employees,
                                                                            self.total_quantity,
                                                                            self.address,
                                                                            self.region_id)


class ConstructionFirm(Firm):
    type = 'CONSTRUCTION'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.houses_built = []
        self.houses_for_sale = []
        self.building = defaultdict(dict)
        self.cash_flow = defaultdict(float)
        self.monthly_planned_revenue = list()

    def plan_house(self, regions, houses, params, seed, seed_np, vacancy_prob):
        """Decide where to build with which attributes """
        # Check whether production capacity does not exceed hired construction
        # for the next construction cash flow period
        if self.building:
            # Number of houses being built is endogenously dependent on number of workers and productivity within a
            # parameter-specified number of months.
            if sum([self.building[b]['cost'] for b in self.building]) > params['CONSTRUCTION_ACC_CASH_FLOW'] * \
                    self.total_qualification(params['PRODUCTIVITY_EXPONENT']) / \
                    params['PRODUCTIVITY_MAGNITUDE_DIVISOR']:
                return
            else:
                self.increase_production = True

        # Candidate regions for licenses and check of funds to buy license
        regions = [r for r in regions if r.licenses > 0 and self.total_balance > r.license_price]
        if not regions:
            return

        # Probability depends on size of market
        if vacancy_prob:
            if seed.random() > vacancy_prob:
                return

        # Targets
        building_size = seed.lognormvariate(4.96, .5)
        b, c, d = .38, .3, .1
        building_quality = seed_np.choice([1, 2, 3, 4], p=[1 - (b + c + d), b, c, d])

        # Get information about region house prices
        region_ids = [r.id for r in regions]
        region_prices = defaultdict(list)
        for h in houses:
            # In correct region
            # within 40 size units,
            # within 2 quality
            if h.region_id in region_ids\
                    and abs(h.size - building_size) <= 40 \
                    and abs(h.quality - building_quality) <= 2:
                region_prices[h.region_id].append(h.price)
                # Only take a sample
                if len(region_prices[h.region_id]) > 100:
                    region_ids.remove(h.region_id)
                    if not region_ids:
                        break

        # Number of product quantities needed for the house
        gross_cost = building_size * building_quality
        # Productivity of the company may vary double than exogenous set markup.
        # Productivity reduces the cost of construction and sets the size of profiting when selling
        productivity = seed.randint(100 - int(2 * params['MARKUP'] * 100), 100) / 100
        building_cost = gross_cost * productivity

        # Choose region where construction is most profitable
        # There might not be samples for all regions, so fallback to price of 0
        region_mean_prices = {r_id: sum(vs)/len(vs) for r_id, vs in region_prices.items()}
        region_profitability = [region_mean_prices.get(r.id, 0) - (r.license_price * building_cost *
                                                                   (1 + params['LOT_COST']))
                                for r in regions]
        regions = [(r, p) for r, p in zip(regions, region_profitability) if p > 0]

        # No profitable regions
        if not regions:
            return

        # Choose region with the highest profitability
        region = max(regions, key=lambda rp: rp[1])[0]
        idx = max(self.building) + 1 if self.building else 0
        self.building[idx]['region'] = region.id
        self.building[idx]['size'] = building_size
        self.building[idx]['quality'] = building_quality

        # Product.quantity increases as construction moves forward and is deducted at once
        self.building[idx]['cost'] = building_cost * region.license_price

        # Provide temporary cashflow revenue numbers before sales start to trickle in.
        # Additional value per month. Expectations of monthly payments before first sell
        self.monthly_planned_revenue.append(self.building[idx]['cost'] / params['CONSTRUCTION_ACC_CASH_FLOW'])

        # Buy license
        region.licenses -= 1
        # Region license price is current QLI. Lot price is the model parameter
        cost_of_land = region.license_price * building_cost * params['LOT_COST']
        self.total_balance -= cost_of_land
        region.collect_taxes(cost_of_land, 'transaction')

    def build_house(self, regions, generator):
        """Firm decides if house is finished"""
        if not self.building:
            return

        # Not finished
        min_cost_idx = [k for k in self.building if self.building[k]['cost'] < self.total_quantity]
        if not min_cost_idx:
            return
        else:
            # Choose the house that entered earlier and for which there is enough material to instantaneously build.
            min_cost_idx = min(min_cost_idx)

        # Finished, expend inputs
        # Remember: if inventory of products is expanded for more than 1, this needs adapting
        building_info = self.building[min_cost_idx]
        paid = min(building_info['cost'], self.inventory[0].quantity)
        self.inventory[0].quantity -= paid

        # Choose random place in region
        region = regions[building_info['region']]
        probability_urban = generator.seed_np.choice([True, False],
                                                     p=[generator.prob_urban(region),
                                                        (1 - generator.prob_urban(region))])
        if probability_urban:
            address = generator.get_random_points_in_polygon(generator.urban[region.id[:7]])[0]
        else:
            address = generator.get_random_points_in_polygon(region)[0]
        # Create the house
        house_id = generator.gen_id()
        size = building_info['size']
        quality = building_info['quality']
        price = (size * quality) * region.index
        h = House(house_id, address, size, price, region.id, quality, owner_id=self.id, owner_type=House.Owner.FIRM)

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
                self.cash_flow[date] += amount/acc_months
                date += relativedelta.relativedelta(months=+1)

    def wage_base(self, unemployment, relevance_unemployment):
        self.revenue = self.cash_flow[self.present]
        # Using temporary planned income before money starts to flow in
        if self.revenue == 0 and self.monthly_planned_revenue:
            # Adding the last planned house income
            self.revenue = self.monthly_planned_revenue[-1]
        # Observing global economic performance has the added advantage of not spending all revenue on salaries
        if self.num_employees > 0:
            return (self.revenue / self.num_employees) * (1 - (unemployment * relevance_unemployment))
        else:
            return self.revenue * (1 - (unemployment * relevance_unemployment))

    def decision_on_prices_production(self, sticky_prices, markup, seed, avg_prices,
                                      prod_exponent=None, prod_magnitude_divisor=None, const_cash_flow=None,
                                      price_ruggedness=None):
        """ Update signal for the labor market
        """
        if seed.random() > sticky_prices:
            if self.building:
                # Number of houses being built is endogenously dependent on number of workers and productivity within a
                # parameter-specified number of months.
                if sum([self.building[b]['cost'] for b in self.building]) > const_cash_flow * \
                        self.total_qualification(prod_exponent) / prod_magnitude_divisor:
                    self.increase_production = True
            else:
                self.increase_production = False

    @property
    def n_houses_sold(self):
        return len(self.houses_built) - len(self.houses_for_sale)

    def mean_house_price(self):
        if not self.houses_built:
            return 0
        t = sum(h.price for h in self.houses_built)
        return t/len(self.houses_built)
