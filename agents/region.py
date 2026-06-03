import json
from shapely.geometry import shape
from collections import defaultdict


class Region:
    """Collects taxes and applies to ameliorate quality of life"""

    def __init__(self, region, index=1, gdp=0, pop=0, total_commute=0, licenses=0):
        # A region is a Geopandas object that contains
        self.address_envelope = region.total_bounds
        self.addresses = region.geometry.unary_union
        self.id = str(region.reset_index().loc[0, 'id'])
        self.index = index
        self.gdp = gdp
        self.pop = pop
        self.licenses = licenses
        self.total_commute = total_commute
        self.cumulative_treasure = defaultdict(int)
        self.treasure = defaultdict(int)
        self.applied_treasure = defaultdict(int)
        self.registry = defaultdict(list)

    @property
    def license_price(self):
        return self.index

    @property
    def total_treasure(self):
        return sum(self.treasure.values())

    def collect_taxes(self, amount, key):
        self.treasure[key] += amount

    def save_and_clear_treasure(self):
        for key in self.treasure.keys():
            if key == 'emissions':
                # Keep emissions treasure persistent so subsidy gate can check it
                continue
            self.cumulative_treasure[key] += self.treasure[key]
            self.treasure[key] = 0

    def transfer_treasure(self):
        treasure = self.treasure.copy()
        self.save_and_clear_treasure()
        return treasure

    def update_qli(self, gdp_per_pop, params):
        """Logistic growth toward QLI_MAX driven by economic development level.

        delta = QLI_GROWTH_RATE × sqrt(gdp_per_pop / QLI_GDP_NORM) × (1 − index / QLI_MAX)

        - sqrt dampens the spread between rich and poor cities (4× GDP gap → 2× rate gap)
        - logistic ceiling: growth slows as index approaches QLI_MAX; never overshoots
        - scale-free: same formula for any city size at the same development level
        - Replaces both the additive tax channel and the multiplicative population channel
        """
        gdp_norm = max(params['QLI_GDP_NORM'], 1e-6)
        economic_driver = (gdp_per_pop / gdp_norm) ** 0.5
        logistic_ceiling = max(0.0, 1.0 - self.index / params['QLI_MAX'])
        self.index += params['QLI_GROWTH_RATE'] * economic_driver * logistic_ceiling

    def update_applied_taxes(self, amount, key):
        self.applied_treasure[key] += amount

    def update_index(self, value):
        """Kept for backward compatibility; no longer called by the main loop."""
        self.index += value

    def update_index_pop(self, proportion_pop, elasticity):
        """Kept for backward compatibility; no longer called by the main loop."""
        self.index *= proportion_pop ** elasticity

    def __repr__(self):
        return '%s \n QLI: %.2f, \t GDP: %.2f, \t Pop: %s, Commute: %.2f' % (self.name, self.index, self.gdp,
                                                                             self.pop, self.total_commute)