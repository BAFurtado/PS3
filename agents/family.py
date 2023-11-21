import datetime

import numpy as np


class Family:
    """
    Family class. Nothing but a bundle of Agents together.
    Generated once and fixed.
    Families share resources equally and move together from household to household.
    Children, when born, remain inside the same family.

    - Setup family class
    - Relevant to distribute income among family members
    - Mobile, as it changes houses when in the housing market

    # Families money:
    1. balance is money just received from a month's salary of all members
    2. savings is a short-time money kept within the family.
    3. when savings exceed a six-month amount, it is deposited to perceive interest (central.wallet)
    """

    def __init__(self, _id,
                 balance=0,
                 savings=0,
                 house=None):
        self.space_constraint = 0
        self.quality_score = 0
        self.bank_savings = 0
        self.probability_employed = 0
        self.have_loan = None
        self.id = _id
        self.balance = balance
        self.savings = savings
        self.owned_houses = list()
        self.members = {}
        self.relatives = set()
        # Refers to the house the family is living on currently
        self.house = house
        self.rent_default = 0
        self.rent_voucher = 0
        self.average_utility = 0
        self.last_permanent_income = list()

        # Previous region id
        if house is not None:
            self.region_id = house.region_id
        else:
            self.region_id = None

    def add_agent(self, agent):
        """Adds a new agent to the set"""
        self.members[agent.id] = agent
        agent.family = self

    def remove_agent(self, agent):
        agent.family = None
        del self.members[agent.id]

    def move_in(self, house):
        if house.family_id is None:
            self.house = house
            house.family_id = self.id
            self.region_id = house.region_id
        else:
            raise Exception

    def move_out(self, funds):
        # If family still has policy money for rent payment and is moving out, give back the money to municipality
        if self.house.rent_data:
            if self.rent_voucher:
                funds.policy_money[self.region_id[:7]] += self.rent_voucher * self.house.rent_data[0]
                self.rent_voucher = 0
        self.house.empty()
        self.house = None
        self.region_id = None

    @property
    def address(self):
        if self.house is not None:
            return self.house.address

    # Budget operations ##############################################################################################
    def get_total_balance(self):
        """Calculates the total available balance of the family"""
        self.balance = sum(m.money for m in self.members.values())
        return self.balance

    def update_balance(self, amount):
        """Evenly distribute money to each member"""
        if self.members:
            per_member = amount / self.num_members
            for member in self.members.values():
                member.money += per_member

    def grab_savings(self, bank, y, m):
        """Withdraws total available balance of the family"""
        s = self.savings
        self.savings = 0
        s += bank.withdraw(self, y, m)
        return s

    def get_wealth(self, bank):
        """ Calculate current wealth, including real estate, debts, and bank savings. """
        estate_value = sum(h.price for h in self.owned_houses)
        # Returns a list of loan objects--which there is just one
        self.have_loan = bank.loans.get(self.id)
        return self.savings + estate_value + bank.sum_deposits(self) - bank.loan_balance(self.id)

    def invest(self, bank, y, m):
        # Savings are updated during consumption as the fraction of above permanent income that is not consumed
        # If savings are above a six-month period reserve money, the surplus is invested in the bank.
        reserve_money = self.get_permanent_income() * 6
        if self.savings > reserve_money > 0:
            bank.deposit(self, self.savings - reserve_money, datetime.date(y, m, 1))
            self.savings = reserve_money

    def total_wage(self):
        return sum(member.last_wage for member in self.members.values() if member.last_wage is not None)

    def get_permanent_income(self):
        return sum(self.last_permanent_income) / len(self.last_permanent_income) if self.last_permanent_income else 0

    def permanent_income(self, bank, r):
        # Equals Consumption (Bielefeld, 2018, pp.13-14)
        # Using last wage available as base for permanent income calculus: total_wage = Human Capital
        t0 = self.total_wage()
        r_1_r = r / (1 + r)
        # Calculated as "discounted sum of current income and expected future income" plus "financial wealth"
        # Perpetuity of income is a fraction (r_1_r) of income t0 divided by interest r
        self.last_permanent_income.append(r_1_r * t0 + r_1_r * (t0 / r) + self.get_wealth(bank) * r)
        return self.get_permanent_income()

    def prob_employed(self):
        """Proportion of members that are employed"""
        employable = [m for m in self.members.values() if 16 < m.age < 70]
        # Employed among those within age to be employed
        self.probability_employed = len([m for m in employable if m.firm_id is not None])/len(employable) \
            if employable else 0
        return self.probability_employed

    def get_prob_employed(self):
        # To avoid calculating it twice in a month
        return self.probability_employed

    # Consumption ####################################################################################################
    def decision_enter_house_market(self, sim, house_price_quantiles):
        # In construction adding criteria: affordability, housing needs (renting), estability (jobs), space constraints?
        # 0. If family has not made goods consumption, or is defaulting on rent don't consider entering housing market
        if not self.average_utility or self.rent_default:
            return False
        # 1. Needs to have short term reserve money
        if not self.savings:
            return False
        # 2. Needs to have some investment in the bank
        self.bank_savings = sim.central.sum_deposits(self)
        if not self.bank_savings:
            return False
        # Distinction on submarket. How much money available compared to housing prices distribution?
        available = self.savings + self.bank_savings
        self.quality_score = np.searchsorted(house_price_quantiles, available)
        # B. How many are employed?
        prob_employed = self.prob_employed()
        # C. Is renting
        # D. Space constraint
        self.space_constraint = self.num_members / self.house.size * 3.5  # To approximate value to a range 0, 1
        return self.is_renting + prob_employed + self.space_constraint

    def decision_on_consumption(self, central, r, year, month):
        """ Family consumes its permanent income, based on members' wages, real estate assets, and savings.
        A. Separate expenses for renting, goods' consumption, education, banking loans, and investments in that order.
         """
        # 1. Grabs wages, money in wallet, from family members.
        money = sum(m.grab_money() for m in self.members.values())
        # 2. Calculate permanent income
        permanent_income = self.permanent_income(central, r)
        # Having loans will impact on a lower long-run permanent income consumption and on a monthly reduction on
        # consumption. However, the price of the house may be appreciating in the market.
        # 3. Total spending equals permanent income.
        # 4. Total spending equals rent (if it is the case), education, loans, consumption.
        rent, loan, consumption = 0, 0, 0
        if self.is_renting and not self.rent_voucher:
            rent = self.house.rent_data[0]
        if self.have_loan:
            # Reserve at least the amount for the due monthly payment
            if self.have_loan[0].age < len(self.have_loan[0].payment):
                loan = self.have_loan[0].payment[self.have_loan[0].age]
            else:
                # Loan age exceeded. Will make payments only when there is enough money, after consumption.
                loan = min(self.have_loan[0].payment)

        # Guard the cases that family expenses exceed resources
        if money >= permanent_income:
            consumption = permanent_income - rent - loan
        # Getting extra funds
        else:
            # If not enough, grab reserve money, savings which are not in the bank.
            money += self.savings
            self.savings = 0
            if money >= permanent_income - rent - loan:
                consumption = permanent_income - rent - loan
            else:
                # If still not enough, grab actual savings in the bank.
                if central.wallet[self]:
                    money += self.grab_savings(central, year, month)
                    if money >= permanent_income - rent - loan:
                        consumption = permanent_income - rent - loan
                    else:
                        consumption = 0

        # If we grabed more than planned
        if money > consumption + rent + loan:
            # Deposit money above that of expenses
            self.savings += (money - consumption)
        return consumption

    def consume(self, regional_market, firms, central, regions, params, seed, year, month, if_origin):
        """Consumption from goods and services firms, based on criteria of price or distance.
        Family general consumption depends on its permanent income, based on members wages, working life expectancy
        and real estate and savings interest
        """
        money_to_spend = self.decision_on_consumption(central, central.interest, year, month)
        # Decision on how much money to consume or save

        if money_to_spend is not None:
            # Picks SIZE_MARKET number of firms at seed and choose the closest or the cheapest
            # Consumes from each product the chosen firm offers
            self.average_utility = 0
            # Here each sector to buy from are in the rows, and the buying column refer to HouseholdConsumption
            # Construction and Government are 0 in the table. Specific construction market apply
            for sector in regional_market.final_demand.index:
                money_this_sector = money_to_spend * regional_market.final_demand['HouseholdConsumption'][sector]
                # Some sectors have 0 value, such as Government, Mining, and Construction (an explicit market is used)
                if money_this_sector == 0:
                    continue
                # Choose the firm to buy inputs from
                sector_firms = [f for f in firms.values() if f.sector == sector]
                market = seed.sample(sector_firms, min(len(firms), int(params['SIZE_MARKET'])))
                market = [firm for firm in market if firm.get_total_quantity() > 0]
                if market:
                    # Choose between cheapest or closest
                    firm_strategy = seed.choice(['Price', 'Distance'])

                    if firm_strategy == 'Price':
                        # Choose firm with the cheapest average prices
                        chosen_firm = min(market, key=lambda firm: firm.prices)
                    else:
                        # Choose the closest firm
                        chosen_firm = min(market, key=lambda firm: self.house.distance_to_firm(firm))

                    # Buy from chosen company
                    change = chosen_firm.sale(money_this_sector, regions, params['TAX_CONSUMPTION'],
                                              self.region_id, if_origin)
                    self.savings += change

                    # Update monthly family utility
                    self.average_utility += money_this_sector - change

    @property
    def agents(self):
        return list(self.members.values())

    def is_member(self, _id):
        return _id in self.members

    @property
    def num_members(self):
        return len(self.members)

    def __repr__(self):
        return 'Family ID %s, House ID %s, Savings $ %.2f, Balance $ %.2f' % \
            (self.id, self.house.id if self.house else None, self.savings, self.get_total_balance())

    @property
    def is_renting(self):
        return self.house.rent_data is not None
