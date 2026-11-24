"""
This module is where the real estate market takes effect.
Definitions on ownership and actual living residence is made.
"""
from collections import defaultdict
import heapq
import numpy as np
from numpy import median
from copy import deepcopy
from .rentmarket import RentalMarket, collect_rent
from world.location import regional_vacancy
from agents.house import House


class HousingMarket:
    def __init__(self):
        self.rental = RentalMarket()
        self.for_sale = set()
        self.policy_year = 0
        self.policy_percentages = defaultdict(dict)

    @staticmethod
    def process_monthly_rent(sim):
        """ Collection of rental payment due made by households that are renting """
        to_pay_rent = [h for h in sim.houses.values() if h.rent_data is not None]
        sim.rent_received = collect_rent(to_pay_rent, sim)

    def update_for_sale(self, sim):
        # Using neighborhood_wealth effect as attractive for prices
        neighborhood_wealth = dict()
        if sim.PARAMS['NEIGHBORHOOD_EFFECT']:
            # Single pass: group incomes by region, then compute medians
            region_incomes = defaultdict(list)
            for f in sim.families.values():
                region_incomes[f.house.region_id].append(f.permanent_income)
            neighborhood_wealth = {key: median(incomes) for key, incomes in region_incomes.items()}
            del region_incomes
            _max, _min = max(neighborhood_wealth.values()), min(neighborhood_wealth.values())
            neighborhood_wealth = {k: (v - _min) / (_max - _min) for k, v in neighborhood_wealth.items()}

        try:
            vacancy = sim.stats.vacancy_rate
        except AttributeError:
            vacancy = 0
        # VACANCY_SCOPE 'region': each house's price responds to its region's vacancy, shrunk toward its municipality's
        regional = None
        if sim.PARAMS.get('VACANCY_SCOPE', 'acp') == 'region':
            regional = regional_vacancy(sim.houses.values())
        if sim.PARAMS.get('BUILD_VACANCY_SCOPE', 'acp') == 'region':
            # Read by ConstructionFirm.plan_house next month
            sim.build_vacancy = regional if regional is not None else regional_vacancy(sim.houses.values())
        if sim.PARAMS.get('LAND_RULE', 'share') == 'residual':
            # Read by ConstructionFirm.plan_house next month
            House.set_land_index(sim.houses.values(), sim.regions)

        occupy = sim.PARAMS.get('HOUSING_PURCHASE', 'occupy') == 'occupy'
        for house in sim.houses.values():
            # Updating all houses values every month
            house.update_price(sim.regions,
                               sim.PARAMS['ON_MARKET_DECAY_FACTOR'],
                               sim.PARAMS['MAX_OFFER_DISCOUNT'],
                               neighborhood_wealth,
                               sim.PARAMS['NEIGHBORHOOD_EFFECT'],
                               vacancy=vacancy if regional is None else regional[house.region_id],
                               offer_size=sim.PARAMS['OFFER_SIZE_ON_PRICE'],
                               vacancy_ref=sim.PARAMS['VACANCY_PRICE_REFERENCE'],
                               max_disc=sim.PARAMS['MAX_OFFER_DISCOUNT'],
                               max_prem=sim.PARAMS['MAX_OFFER_PREMIUM'])

            # If house is empty, and not already on sales list, add it to houses on the market and start counting
            # However, if house is empty and had been empty count one extra month
            if not house.is_occupied:
                if house not in self.for_sale:
                    house.on_market = 0
                    self.for_sale.add(house)
                else:
                    house.on_market += 1
            elif occupy:
                # An occupied house is not for sale
                self.for_sale.discard(house)

        # Order houses by price most expensive first
        # self.for_sale.sort(key=lambda h: h.price, reverse=True)

    def housing_market(self, sim, house_price_quantiles):
        """Start of the housing market"""
        # Endogenously select families entering market with a threshold set parameter
        # Criteria (family.py) include space, renting, jobs, enough savings
        threshold = round(len(sim.families) * sim.PARAMS['PERCENTAGE_ENTERING_ESTATE_MARKET'])
        # Looking now is a tuple with family in position 0 and score of entering the house market 1

        scored = []
        for f in sim.families.values():
            score = f.decision_enter_house_market(sim, house_price_quantiles)
            if score > 0:
                scored.append((score, f))

        looking = [
            f for score, f in heapq.nlargest(
                threshold,
                scored,
                key=lambda x: x[0]
            )
        ]
        regions = np.unique([int(region[0:6]) for region in sim.regions.keys()])
        if sim.clock.year > self.policy_year:
            self.policy_year = sim.clock.year
            for r in regions:
                self.policy_percentages[(sim.clock.year, r)] = deepcopy(sim.central.funding[(sim.clock.year, r)])
        # Update funding
        for r in regions:
            for loan_type in ['recursos_fgts', 'recursos_sbpe',]:
                value = max(0,
                            self.policy_percentages[(sim.clock.year, r)][loan_type]
                            * sim.stats.funds_gdp(r, sim.clock.year))
                sim.central.funding[(sim.clock.year, r)][loan_type] = value
                sim.central.monthly_funding_available[(sim.clock.year, sim.clock.months, r, loan_type)] = value

        # Update prices of all houses in the simulation and status 'on_market' or not
        self.update_for_sale(sim)

        # Run the market
        self.allocate_houses(sim, looking)

    def allocate_houses(self, sim, looking):
        """ This function manipulates both lists for families that are purchasing or renting
            and houses that are available for rent or purchase.
            Families "looking" for houses become either "purchasing" or "renting"
            Houses are either on "for_sale" or "for_rent"
        """
        # If empty lists, stop procedure
        if not looking or not self.for_sale:
            return

        # Families check the bank for potential credit
        # Check families that have access to subsidised rates
        # First FGTS, then SBPE, then MARKET
        incomes = np.array([f.get_permanent_income() for f in looking])
        fgts_quantiles_value = np.quantile(incomes, sim.PARAMS['FGTS_INCOME_QUANTILE'])
        sbpe_quantiles_value = np.quantile(incomes, sim.PARAMS['SBPE_INCOME_QUANTILE'])

        for f, income in zip(looking, incomes):
            if income < fgts_quantiles_value:
                f.savings_with_loan = f.savings + f.bank_savings + sim.central.max_loan(f, flag='fgts', table_type='price')[0]
                f.loan_rate = 'fgts'
            elif income < sbpe_quantiles_value:
                f.savings_with_loan = f.savings + f.bank_savings + sim.central.max_loan(f, flag='sbpe')[0]
                f.loan_rate = 'sbpe'
            else:
                f.savings_with_loan = f.savings + f.bank_savings + sim.central.max_loan(f)[0]
                f.loan_rate = 'market'

        # Family with the largest savings
        family_maximum_purchasing_power = max(looking, key=lambda fam: fam.savings_with_loan)
        maximum_purchasing_power = family_maximum_purchasing_power.savings_with_loan

        # Sampling below draws from these lists, so their order must be stable.
        for_sale_ordered = sorted(self.for_sale, key=lambda h: h.id)
        # Only rent from families, not firms
        family_houses_to_rent = [h for h in for_sale_ordered if h.family_owner]

        houses_to_rent = sim.seed.sample(family_houses_to_rent,
                                         int(len(family_houses_to_rent) * sim.PARAMS['INITIAL_RENTAL_SHARE']))

        # Deduce houses that are to be rented from sales pool and
        # Restrict list of available houses to families' maximum paying ability
        houses_to_rent_set = set(houses_to_rent)
        houses_to_sell = [h for h in for_sale_ordered if
                          (h not in houses_to_rent_set) and (h.price < maximum_purchasing_power)]

        # Separating purchasing families by house price distribution and assigning quality houses accordingly
        purchasing, houses_to_sell_by_quality, houses_to_rent_by_quality = \
            defaultdict(list), defaultdict(list), defaultdict(list)

        for family in looking:
            # House quality varies from 1 to 4, Quartiles of income varies from 0 to 3. Adjusting to be compatible.
            purchasing[family.quality_score + 1].append(family)
        for house in houses_to_sell:
            houses_to_sell_by_quality[house.quality].append(house)
        for house in houses_to_rent:
            houses_to_rent_by_quality[house.quality].append(house)

        # Create two (local) lists for those families that are Purchasing and those that are Renting
        if not houses_to_sell_by_quality:
            # Obviously, if there are no houses for sale, all unoccupied are for rentals.
            renting = purchasing
        else:
            # Manipulating lists. Getting list by sampling and by condition
            # Renting families is a share of those moving. Both rich and poor may rent.
            # Rationale for decision on renting in the literature is dependent on loads of future uncertainties.
            renting, willing = defaultdict(list), defaultdict(list)
            for quality_key in purchasing:
                renting[quality_key] = sim.seed.sample(purchasing[quality_key],
                                                       int(len(purchasing[quality_key]) *
                                                       sim.PARAMS['INITIAL_RENTAL_SHARE']))

                # The families that are not renting, want to join the purchasing list
                willing[quality_key] = [f for f in purchasing[quality_key] if f not in renting[quality_key]]

                # Minimum price on market
                # Families that cannot afford to buy, will also have to join the renting list...
                if houses_to_sell_by_quality[quality_key]:
                    house_minimum_price = min(houses_to_sell_by_quality[quality_key], key=lambda h: h.price)
                    minimum_price = house_minimum_price.price
                    [renting[quality_key].append(f) for f in willing[quality_key] if
                     f.savings_with_loan < minimum_price]

                # ... and only those who remain will join the purchasing list
                purchasing[quality_key] = [f for f in willing[quality_key] if f not in renting[quality_key]]

        # Call Rental market ###############################################################
        for quality_key in houses_to_rent_by_quality:
            if renting[quality_key] and houses_to_rent_by_quality[quality_key]:
                self.rental.rental_market(renting[quality_key], sim, houses_to_rent_by_quality[quality_key])

        # Second check. If empty lists, stop procedure
        if not purchasing or not houses_to_sell_by_quality:
            return

        # Proceed to Sales market ###########################################################
        vacancy = sim.stats.vacancy_rate

        # HOUSING_BIDDING 'filatova': buyers of a submarket bid against each other in rounds
        if sim.PARAMS.get('HOUSING_BIDDING', 'off') == 'filatova':
            for quality_key in purchasing:
                if purchasing[quality_key] and houses_to_sell_by_quality[quality_key]:
                    self.bidding(purchasing[quality_key], houses_to_sell_by_quality[quality_key], sim, vacancy)
            return

        # For each family
        for quality_key in purchasing:
            for family in purchasing[quality_key]:
                # Sending a smaller list to negotiate within specific submarket
                # (same quality as household savings distribution quantile)
                self.negotiating(family, houses_to_sell_by_quality[quality_key], sim, vacancy)

    def negotiating(self, family, for_sale, sim, vacancy):
        my_market = sim.seed.sample(for_sale, min(len(for_sale), int(sim.PARAMS['SIZE_MARKET']) * 3))
        my_market.sort(key=lambda h: h.price, reverse=True)
        # If family has enough funds, or successfully gets a loan, it buys the first house of the stack.
        # Only houses that are within savings or savings plus loan compose each family individual market
        # Otherwise, it tries another one.
        terms = self.terms(family, sim)
        savings, savings_with_mortgage, capped_low_value, occupy = terms[0], terms[1], terms[3], terms[6]
        location = getattr(sim, 'location_choice', None)
        if location is not None and location.ready:
            # LOCATION_CHOICE 'logit': the family picks among the houses it can pay for; a failed purchase drops
            # the house and the family picks again
            options = [h for h in my_market if not (occupy and h.is_occupied)
                       and (savings > h.price or savings_with_mortgage > h.price
                            or savings / h.price > capped_low_value)]
            while options:
                house = location.choose(family, options)
                if self.buy(family, house, for_sale, sim, vacancy, terms):
                    return
                options.remove(house)
            return
        for house in my_market:
            # A house can be occupied since the list was drawn up this month
            if occupy and house.is_occupied:
                continue
            if self.buy(family, house, for_sale, sim, vacancy, terms):
                return

    @staticmethod
    def tightness(buyers, sellers):
        """Filatova et al. (2009, eq. 6): (NB - NS) / (NB + NS), at least zero"""
        return max(0.0, (buyers - sellers) / (buyers + sellers)) if buyers + sellers else 0.0

    @staticmethod
    def bid(house, terms):
        """The family's bid for `house`: its means (savings, else savings plus the loan) capped at terms[2] times
        the ask, or its savings when they reach only CAPPED_LOW_VALUE of the ask. None when it cannot bid.
        buy() prices a bid at or above the ask at (ask + bid) / 2"""
        savings, savings_with_mortgage, capped_top_value, capped_low_value = terms[:4]
        p = house.price
        if savings > p:
            return min(savings, capped_top_value * p)
        if savings_with_mortgage > p:
            return min(savings_with_mortgage, capped_top_value * p)
        if savings / p > capped_low_value:
            return savings
        return None

    def bidding(self, families, for_sale, sim, vacancy):
        """HOUSING_BIDDING 'filatova' (Filatova et al. 2009, p.7): in each round every family still looking bids on
        its preferred house among those it can bid on (the dearest, or its LOCATION_CHOICE pick), and each house goes
        to its highest bid: the bidders try buy() from the highest bid down (ties in random order) until one buys.
        A family that does not buy drops that house and bids again next round. Rounds end when no family has a
        house left to bid on. Bids are capped at the ask x (1 + ε), ε = tightness(buyers, houses) of the submarket at
        the start of the month (eq. 6), in place of CAPPED_TOP_VALUE: a buyer pays from the ask, when houses
        outnumber buyers, to 1.5 x the ask"""
        cap = 1 + self.tightness(len(families), len(for_sale))
        location = getattr(sim, 'location_choice', None)
        if location is not None and not location.ready:
            location = None
        markets = {}
        for family in families:
            my_market = sim.seed.sample(for_sale, min(len(for_sale), int(sim.PARAMS['SIZE_MARKET']) * 3))
            my_market.sort(key=lambda h: h.price, reverse=True)
            t = self.terms(family, sim)
            markets[id(family)] = (t[:2] + (cap, t[3], (cap + 1) / 2) + t[5:], my_market)
        sold = set()
        active = list(families)
        while active:
            bids, houses = defaultdict(list), {}
            for family in active:
                terms, my_market = markets[id(family)]
                occupy = terms[6]
                options = [h for h in my_market if h.id not in sold and not (occupy and h.is_occupied)
                           and self.bid(h, terms) is not None]
                if not options:
                    continue
                house = location.choose(family, options) if location is not None else options[0]
                houses[house.id] = house
                bids[house.id].append((self.bid(house, terms), sim.seed.random(), id(family), family))
            if not bids:
                return
            bought = set()
            for house_id in sorted(bids):
                house = houses[house_id]
                for *_, family in sorted(bids[house_id], key=lambda b: (-b[0], b[1])):
                    if self.buy(family, house, for_sale, sim, vacancy, markets[id(family)][0]):
                        sold.add(house_id)
                        bought.add(id(family))
                        break
                    markets[id(family)][1].remove(house)
                for *_, family in bids[house_id]:
                    if id(family) not in bought and house in markets[id(family)][1]:
                        markets[id(family)][1].remove(house)
            active = [f for f in active if id(f) not in bought]

    @staticmethod
    def terms(family, sim):
        """The family's means and the market rules buy() applies to it"""
        savings = family.savings + family.bank_savings
        savings_with_mortgage = family.savings_with_loan
        # Cache parameters locally (avoid repeated dict lookups)
        params = sim.PARAMS
        capped_top_value = params['CAPPED_TOP_VALUE']
        capped_low_value = params['CAPPED_LOW_VALUE']
        # Offer ceiling. The bargained price averages the buyer's means with the
        # asking price; when the buyer's means are capped at CAPPED_TOP_VALUE times
        # the ask, that average is (CAPPED_TOP_VALUE + 1) / 2 times the ask. The
        # rule is therefore continuous at the cap: at savings/p == 1.3 both
        # branches give 1.15p.
        capped_price_factor = (capped_top_value + 1) / 2
        loan_rate = getattr(family, 'loan_rate', None)
        if loan_rate == 'fgts':
            max_loan_to_value = params['MAX_LOAN_TO_VALUE_FGTS']
        elif loan_rate == 'sbpe':
            max_loan_to_value = params['MAX_LOAN_TO_VALUE_SBPE']
        else:
            max_loan_to_value = params['MAX_LOAN_TO_VALUE']

        occupy = params.get('HOUSING_PURCHASE', 'occupy') == 'occupy'
        return (savings, savings_with_mortgage, capped_top_value, capped_low_value, capped_price_factor,
                max_loan_to_value, occupy)

    def buy(self, family, house, for_sale, sim, vacancy, terms):
        """The family tries to buy `house`. True when bought."""
        (savings, savings_with_mortgage, capped_top_value, capped_low_value, capped_price_factor,
         max_loan_to_value, occupy) = terms
        cash = 0
        # house.price already incorporates the vacancy adjustment via update_price()
        p = house.price

        # If savings is enough, then price is established as the average of the two
        if savings > p:
            # Restrict OFFERs to a maximum of 30% threshold
            if savings / p > capped_top_value:
                price = p * capped_price_factor
            else:
                price = (savings + p) / 2

        # If not, check whether loan can help
        elif savings_with_mortgage > p:
            if savings_with_mortgage / p > capped_top_value:
                price = p * capped_price_factor
            else:
                price = (savings_with_mortgage + p) / 2

            # Get loan to make up the difference. The bank is asked only when a
            # loan is actually needed: a request for zero is not the same thing
            # as no request, and would come back as a refusal.
            loan_amount = max(0, price - savings)
            if loan_amount > 0:
                # Check macroprudencial policy. If loan to value is above set value,
                # no loan, leave the market.
                if (loan_amount / price) > max_loan_to_value:
                    return False

                # Attempt to actually get the loan from the bank
                success, amount = sim.central.request_loan(
                    family, house, loan_amount, sim.clock.year, sim.clock.months
                )
                if not success:
                    return False

                cash += amount

        elif savings / p > capped_low_value:
            if sim.seed_np.rand() < vacancy:
                price = savings
            else:
                return False
        else:
            return False

        # Withdraw the money of buying family from the bank and from savings
        cash += family.grab_savings(sim.central, sim.clock.year, sim.clock.months)
        change = cash - price

        # Register the transaction, collect taxes and consider moving
        self.notarial_procedures(family, house, price, change, sim)

        # if the procedures have come this far, it means loan or price have being agreed upon.
        # Drop the sold house. The rest of the market stays in the set: a house that
        # leaves and is re-added has its on_market counter reset, which suppresses the
        # time-on-market price decay in House.update_price.
        self.for_sale.discard(house)
        if occupy:
            # Later buyers this month draw from the same submarket list
            for_sale.remove(house)

        # Having bought a house, then it can move on to the next family
        return True

    def notarial_procedures(self, family, house, price, change, sim):
        # Withdraw money from buying family and distribute back the difference
        family.update_balance(change)
        sim.house_sales.append((price, house.price, not house.family_owner, house.region_id))
        # Collect taxes on transaction
        taxes = price * sim.PARAMS['TAX_ESTATE_TRANSACTION']
        sim.regions[house.region_id].collect_taxes(taxes, 'transaction')

        if house.family_owner:
            # Deposit money on selling family account
            assert (house.owner_id in sim.firms) is False
            sim.families[house.owner_id].update_balance(price - taxes)
            # Transfer ownership
            sim.families[house.owner_id].owned_houses.remove(house)

        # Firm owner
        else:
            # Deposit money on selling firm
            sim.firms[house.owner_id].update_balance(price - taxes, sim.PARAMS['CONSTRUCTION_ACC_CASH_FLOW'],
                                                     sim.clock.days)
            # Transfer ownership
            sim.firms[house.owner_id].houses_for_sale.remove(house)

        # Finish notarial procedures
        house.owner_id = family.id
        house.family_owner = True
        family.owned_houses.append(house)
        house.on_market = 0

        # Decision on moving
        if family.house is None:
            # If previously without a house, move into the new acquired one
            self.make_move(family, house, sim)
        elif family.is_renting and sim.PARAMS.get('HOUSING_PURCHASE', 'occupy') == 'occupy':
            # A tenant leaves its rental and moves into the house it bought
            self.make_move(family, house, sim)
        else:
            # Check if there is more than one option and choose
            self.decision(family, sim)

    def decision(self, family, sim):
        """A family decides which house to move into"""
        # Options include only empty houses (those ot rented) or currently occupied by the same family.
        options = [h for h in family.owned_houses if (h.family_id is None) or (h.family_id == family.id)]

        # If options is size 1, it means the only owned house is the one the family currently lives in.
        if len(options) > 1:
            # Sort by price, which captures quality, size, and location
            # This puts the cheapest house first
            options.sort(key=lambda h: h.price, reverse=False)
            # If family does not live in the worst house, but nobody is employed, move to the worst house
            prob_employed = family.get_prob_employed()
            if options[0].family_id != family.id and prob_employed == 0:
                self.make_move(family, options[0], sim)

            # Else if they live in the worst house, but at least one member is working, move to a better house
            elif options[0].family_id == family.id and prob_employed > 0:
                self.make_move(family, options[-1], sim)

    @staticmethod
    def make_move(family, house, sim):
        # Make the move
        old_r_id = family.region_id
        if family.house is not None:
            family.move_out(sim.funds)
        family.move_in(house)
        # Only after simulation has begun, it is necessary to update population, not at generation time
        try:
            if sim.mun_pops:
                if old_r_id and family.region_id:
                    for _ in family.agents:
                        sim.update_pop(old_r_id, family.region_id)
        except AttributeError:
            pass
