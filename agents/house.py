from enum import Enum
from math import e


class Owner(Enum):
    FAMILY = 0
    FIRM = 1


class House:
    """Holds the fixed households.
    They may have changing owners and changing occupancy."""
    Owner = Owner
    # Multiplies size x quality x region index in the price (world/house_values.py)
    price_scale = 1.0
    # RENT_LEVEL 'census': the ACP's factor per region id, on top of price_scale
    area_scale = {}
    # RENT_LEVEL 'census_ap': the ACP's factor per region id, against which land takes the region's premium
    land_base = {}
    # LAND_RULE 'residual': the mean region index over all houses, the index of the reference value
    land_index = None

    @staticmethod
    def scale(region_id):
        """Price per unit of size x quality x region index in the region"""
        return House.price_scale * House.area_scale.get(region_id, 1.0)

    @staticmethod
    def set_land_index(houses, regions):
        """LAND_RULE 'residual': the mean over houses of their region's index"""
        indices = [regions[h.region_id].index for h in houses]
        House.land_index = sum(indices) / len(indices) if indices else None

    @staticmethod
    def land_cost(region_id, units, lot, index=None):
        """Land for `units` of size x quality x index: `lot` of the house's value. With land_base (RENT_LEVEL
        'census_ap') the land also takes the region's value above the ACP level, value - (1 - lot) x the value at
        the ACP factor, at least zero, so the builder's margin does not grow with the region's factor.
        LAND_RULE 'residual' (land_index set, `index` the region's): the reference value is the same house at the
        ACP factor and at land_index, so the land also takes the value of the region's index above the mean"""
        if not House.land_base and House.land_index is None:
            return units * House.scale(region_id) * lot
        base = House.price_scale * House.land_base.get(region_id, House.area_scale.get(region_id, 1.0))
        if House.land_index is not None and index:
            return max(0.0, units * House.scale(region_id) - (1 - lot) * units * House.land_index / index * base)
        return max(0.0, units * (House.scale(region_id) - (1 - lot) * base))

    def __init__(self, _id, address, size, price, region_id, quality,
                 rural=False, family_id=None, owner_id=None,
                 owner_type=Owner.FAMILY):
        self.id = _id
        self.address = address
        self.size = size
        self.price = price
        self.region_id = region_id
        self.quality = quality
        self.rural = rural
        # Owner may be the occupant or the house may be vacant or rented
        self.family_id = family_id
        self.owner_id = owner_id
        self.owner_type = owner_type
        self.rent_data = None
        self.on_market = 0
        # Cache firm distances, since houses and firms never change address
        self._firm_distances = {}

    def update_price(self, regions, k, bound, neighborhood, value,
                     vacancy=0, offer_size=0, vacancy_ref=0.08,
                     max_disc=0.6, max_prem=1.3):
        """Compute new price for the house"""
        self.price = self.size * self.quality * regions[self.region_id].index * House.scale(self.region_id)
        # The price before the time-on-market and vacancy terms (world/location.py)
        self.intrinsic = self.price
        # Update for too long in the market
        self.price *= (1 - bound) * e ** (k * self.on_market) + bound
        if neighborhood:
            # If neighborhood is 0 (False), price is unchanged.
            # If 1 (True) or higher, neighborhood effect is the value multiplier which increasingly impacts prices
            self.price *= (1 + value * neighborhood.get(self.region_id, 0))
            self.intrinsic *= (1 + value * neighborhood.get(self.region_id, 0))
        if offer_size:
            # Vacancy adjustment baked into the listed price so that both purchase and rental
            # markets see a vacancy-corrected base price (avoiding double-discounting at negotiation time).
            vf = max(min(1 + (vacancy_ref - vacancy) * offer_size, max_prem), max_disc)
            self.price *= vf

    def empty(self):
        """Remove current family"""
        self.family_id = None
        self.rent_data = None

    @property
    def is_occupied(self):
        return self.family_id is not None

    def pay_property_tax(self, sim):
        # Calculate taxes due of property paid in monthly in twelve installments
        tax = self.price * sim.PARAMS['TAX_PROPERTY'] / 12
        # Withdraw from paying family or firm (current occupant or owner when unoccupied)
        # Only paying taxes when money is available
        if self.family_id is None:
            if self.owner_type is House.Owner.FAMILY:
                payer = sim.families[self.owner_id]
            else:
                payer = sim.firms[self.owner_id]
        else:
            payer = sim.families[self.family_id]

        if payer.get_total_balance() > tax:
            payer.update_balance(-tax)

            # Transfer to region
            sim.regions[self.region_id].collect_taxes(tax, 'property')

    def __repr__(self):
        return 'House ID %s, Family ID %s, Owner ID %s, Size %s, Price$ %s, Region %s' % (self.id, self.family_id,
                                                                                          self.owner_id, self.size,
                                                                                          self.price, self.region_id)

    def distance_to_firm(self, firm):
        if firm.id not in self._firm_distances:
            self._firm_distances[firm.id] = self.calculate_distance(firm.address)
        return self._firm_distances[firm.id]

    def calculate_distance(self, address):
        return self.address.distance(address)

    @property
    def family_owner(self):
        return self.owner_type is Owner.FAMILY

    @family_owner.setter
    def family_owner(self, bool):
        if bool:
            self.owner_type = Owner.FAMILY
        else:
            self.owner_type = Owner.FIRM
