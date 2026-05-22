import datetime

import numpy as np
from collections import defaultdict, deque


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
        self.crowding_ratio = 0
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
        self.loan_rate = 'market'
        # Refers to the house the family is living on currently
        self.house = house
        self.rent_default = 0
        self.rent_voucher = 0
        self.average_utility = 0
        self.last_permanent_window = 24
        self.last_permanent_income = deque(maxlen=self.last_permanent_window)
        self.permanent_income = 1
        self.affordability_ratio = 10e6

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
        if bank.wallet.get(self):
            self.bank_savings = 0
            s += bank.withdraw(self, y, m)
        return s

    def get_wealth(self, bank):
        """ Calculate current wealth, including real estate, debts, and bank savings. """
        estate_value = sum(h.price for h in self.owned_houses)
        # Returns a list of loan objects--which there is just one
        self.have_loan = bank.loans.get(self.id)
        return self.savings + estate_value + bank.sum_deposits(self) - bank.loan_balance(self.id)

    def invest(self, bank, y, m, params):
        # Keep a liquid emergency buffer equal to SAVINGS_BUFFER_MONTHS of current wages.
        # Anchored to wages (not permanent income) so it covers actual cash expenses
        # during unemployment and does not compound with house appreciation or interest.
        reserve_money = self.total_wage() * params['SAVINGS_BUFFER_MONTHS']
        if self.savings > reserve_money > 0:
            bank.deposit(self, self.savings - reserve_money, datetime.date(y, m, 1))
            self.savings = reserve_money

    def total_wage(self):
        return sum(member.last_wage for member in self.members.values() if member.last_wage is not None)

    def get_permanent_income(self):
        return self.permanent_income

    def update_affordability(self):
        # Update affordability only
        if self.permanent_income > 0 and self.house:
            self.affordability_ratio = self.house.price / self.permanent_income

    def update_permanent_income(self, bank, r):
        t0 = self.total_wage()
        EPS = 1e-4
        r_eff = max(r, EPS)
        wealth = self.get_wealth(bank)
        # Permanent income (Dawid-consistent)
        current_pi = t0 + r_eff * wealth
        self.last_permanent_income.append(current_pi)
        # deque(maxlen=24) evicts oldest automatically; no manual trimming needed
        # Average without artificial zerollasts
        value = (
            current_pi if len(self.last_permanent_income) == 1
            else np.mean(self.last_permanent_income)
        )
        self.permanent_income = value
        self.update_affordability()
        return value

    def prob_employed(self):
        """Proportion of members that are employed"""
        employable = [m for m in self.members.values() if 16 < m.age < 70]
        # Employed among those within age to be employed
        self.probability_employed = len([m for m in employable if m.firm_id is not None]) / len(employable) \
            if employable else 0
        return self.probability_employed

    def get_prob_employed(self):
        # To avoid calculating it twice in a month
        return self.probability_employed

    # Consumption ####################################################################################################
    def decision_enter_house_market(self, sim, house_price_quantiles):
        """Decide whether to enter the housing market as a potential buyer.

        Returns False to exclude the family, or a score > 0 used to rank families;
        the simulation selects the top PERCENTAGE_ENTERING_ESTATE_MARKET scorers.

        Gates (hard — return False if any fails):
          0. Basic viability: has consumed goods and is not in rent default.
          1. Short-term liquidity: holds some savings.
          2. Active loan: bank will not approve a second mortgage.
          3. Bank deposits: invested surplus signals financial stability.
          4. Down-payment: available funds cover MIN_DOWN_PAYMENT_FRACTION of target.
          5. Employment: prob_employed < 0.3 — too unstable to service a mortgage.

        Score = housing_need − opportunity_cost − liquidity_penalty

          housing_need    ∈ [0, 2]
            is_renting: 1.0 — paying rent is a direct cash drain that ownership ends.
            crowding_bonus ∈ [0, 1]: extra incentive when cramped. Normalized so that
              occupying 1× the house's size scores 0, 3× scores 1.0. Crowding is an
              incentive, never a gate — very crowded owners still try to upgrade.

          opportunity_cost ≥ 0
            max(0, bank_rate − INITIAL_RENTAL_PRICE) × HOUSING_FINANCIAL_WEIGHT
            Dominant disincentive for non-renting owners: when the bank pays more than
            ownership saves, speculative buying is irrational. Scales with the SELIC
            cycle — low rates open the door for some owners; high rates shut it.

          liquidity_penalty ∈ [0, 1]
            Discourages families from locking all savings into a down payment.
            Zero when ≥ LIQUIDITY_BUFFER_MONTHS of income remain liquid after buying.
        """
        # 0. Basic viability
        if not self.average_utility or self.rent_default:
            return False
        # 1. Short-term liquidity
        if not self.savings:
            return False
        # 2. Active loan: bank will not approve a second mortgage;
        #    also prevents loan-holders from crowding out first-time buyers.
        if sim.central.loans[self.id]:
            return False
        # 3. Bank deposits signal financial stability and non-zero surplus
        self.bank_savings = sim.central.sum_deposits(self)
        if not self.bank_savings:
            return False

        # Determine target price submarket from total available funds
        available = self.savings + self.bank_savings
        self.quality_score = np.searchsorted(house_price_quantiles, available)
        target_idx = min(self.quality_score, len(house_price_quantiles) - 1)
        target_price = house_price_quantiles[target_idx]

        # 4. Down-payment gate
        if available < target_price * sim.PARAMS['MIN_DOWN_PAYMENT_FRACTION']:
            return False

        # 5. Employment gate: too unstable to commit to a mortgage
        if self.prob_employed() < 0.3:
            return False

        # --- SCORE ---

        # A. Housing need [0, 2]
        renting_need = 1.0 if self.is_renting else 0.0
        self.crowding_ratio = self.num_members / max(self.house.size, 0.5)
        crowding_bonus = min(1.0, max(0.0, self.crowding_ratio - 1.0) / 2.0)
        housing_need = renting_need + crowding_bonus

        # B. Opportunity cost: bank rate vs rental yield (SELIC-sensitive disincentive)
        EPS = 1e-6
        bank_rate = max(sim.central.interest, EPS)
        rental_yield = sim.PARAMS['INITIAL_RENTAL_PRICE']
        opportunity_cost = max(0.0, bank_rate - rental_yield) * sim.PARAMS['HOUSING_FINANCIAL_WEIGHT']

        # C. Liquidity penalty: discourages depleting all savings in the down payment
        down_payment = target_price * sim.PARAMS['MIN_DOWN_PAYMENT_FRACTION']
        post_purchase_liquid = available - down_payment
        permanent_income = self.get_permanent_income()
        liquidity_months = post_purchase_liquid / max(permanent_income, EPS)
        liquidity_penalty = max(0.0, 1.0 - liquidity_months / sim.PARAMS['LIQUIDITY_BUFFER_MONTHS'])

        return housing_need - opportunity_cost - liquidity_penalty

    def decision_on_consumption(self, central, year, month, params, regions):
        """ Family consumes its permanent income, based on members' wages, real estate assets, and savings.
        A. Separate expenses for renting, banking loans, goods' consumption, and investments in that order.
         """
        # 1. Grabs wages, money in wallet, from family members.
        # This can only be called once due to transport deduction
        money = sum(m.grab_money(params, regions) for m in self.members.values())
        # 2. Calculate permanent income
        permanent_income = self.get_permanent_income()
        # Having loans will impact on a lower long-run permanent income consumption and on a monthly reduction on
        # consumption. However, the price of the house may be appreciating in the market.
        # 3. Total spending equals permanent income.
        # 4. Total spending equals rent (if it is the case), loans, consumption.
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

        propensity = params.get('CONSUMPTION_PROPENSITY', 1.0)
        target = max(0, permanent_income - rent - loan) * propensity

        # Guard the cases that family expenses exceed resources
        if money >= permanent_income:
            consumption = target
        # Getting extra funds
        else:
            # If not enough, grab reserve money, savings which are not in the bank.
            money += self.savings
            self.savings = 0
            if money >= permanent_income - rent - loan:
                consumption = target
            else:
                # If still not enough, grab actual savings in the bank.
                if central.wallet[self]:
                    money += self.grab_savings(central, year, month)
                    if money >= permanent_income - rent - loan:
                        consumption = target
                    else:
                        consumption = max(0, money - rent - loan)

        # If we grabbed more than planned
        if money > consumption + rent + loan:
            # Deposit money above that of expenses
            self.savings += max(0, money - consumption - rent - loan)
        return consumption

    def consume(self, regional_market, seed, seed_np, central, regions, params, year, month,
                if_origin, firms_by_sector):
        """Consumption from goods and services firms, based on criteria of price or distance.
        Family general consumption depends on its permanent income, based on members wages, working life expectancy
        and real estate and savings interest
        """
        # Decision on how much money to consume or save
        money_to_spend = self.decision_on_consumption(central, year, month, params, regions)

        if money_to_spend is None:
            return defaultdict(float)

        size_market = int(params['SIZE_MARKET'])
        tax_consumption = params['TAX_CONSUMPTION']

        household_demand = regional_market.final_demand['HouseholdConsumption']
        total_consumption = defaultdict(float)
        savings = self.savings
        avg_utility = 0.0
        house = self.house
        # Pre-extract house distance cache for inline lookups (avoids repeated method-call overheads)
        house_addr = house.address
        house_dist_cache = house._firm_distances

        for sector, sector_share in household_demand.items():
            if sector_share <= 0:
                continue

            money_this_sector = money_to_spend * sector_share
            if money_this_sector <= 0:
                continue

            sector_firms = firms_by_sector.get(sector)
            if not sector_firms:
                continue

            n_firms = len(sector_firms)

            if n_firms <= size_market:
                market = sector_firms
            else:
                market = seed.sample(sector_firms, size_market)

            if seed.randint(0, 1):
                # Price strategy: direct inventory[0].price access avoids property dispatch
                chosen_firm = min(market, key=lambda f: f.inventory[0].price)
            else:
                # Distance strategy: inline cache lookup avoids per-call method dispatch
                best_firm = None
                best_dist = float('inf')
                for f in market:
                    fid = f.id
                    d = house_dist_cache.get(fid)
                    if d is None:
                        d = house_addr.distance(f.address)
                        house_dist_cache[fid] = d
                    if d < best_dist:
                        best_dist = d
                        best_firm = f
                chosen_firm = best_firm

            change = chosen_firm.sale(
                money_this_sector, regions, tax_consumption,
                self.region_id, if_origin
            )

            savings += change
            utility_gain = money_this_sector - change
            avg_utility += utility_gain
            total_consumption[sector] += utility_gain

        self.savings = savings
        self.average_utility = avg_utility

        return total_consumption

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