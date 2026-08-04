import json
from shapely.geometry import shape
from collections import defaultdict


class Region:
    """Collects taxes and applies to ameliorate quality of life"""

    # Class-level defaults, not merely documentation: Region instances are pickled into
    # `StoragedAgents/*.agents`, which is keyed on GENERATOR_PARAMS with no source hash,
    # so a cache written before these attributes existed is unpickled without them and
    # would otherwise raise on first access. A float default is safe — the first write
    # shadows it with an instance attribute.
    applied_flow = 0.0
    qli_gdp_pc = 0.0
    qli_spend_pc = 0.0

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
        # Monthly FLOW of public money applied in this region. `applied_treasure` is a
        # cumulative stock that is never reset, so it cannot be read as spending in a
        # month; this accumulator is consumed and cleared once per month by
        # Funds.invest_taxes, which is what the fiscal leg of the QLI driver reads.
        self.applied_flow = 0.0
        # Per-capita drivers of the last QLI update, kept for the output writer.
        self.qli_gdp_pc = 0.0
        self.qli_spend_pc = 0.0
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

    def update_qli(self, gdp_per_pop, spend_per_pop, params):
        """Logistic growth toward QLI_MAX driven by economic development level.

        driver = (1 − w) × sqrt(gdp_per_pop / QLI_GDP_NORM)
                 +     w × sqrt(spend_per_pop / QLI_SPEND_NORM)
        delta  = QLI_GROWTH_RATE × driver × (1 − index / QLI_MAX)

        with w = QLI_TAX_WEIGHT. At w = 0 this is exactly the GDP-only rule.

        - sqrt dampens the spread between rich and poor cities (4× GDP gap → 2× rate gap)
        - logistic ceiling: growth slows as index approaches QLI_MAX; never overshoots
        - scale-free: same formula for any city size at the same development level
        - Replaces the multiplicative population channel; the additive tax channel returns
          as the fiscal leg, weighted by w, so public spending can move neighbourhood
          quality and through it house prices. FPM is redistributive, so spend per capita
          moves independently of GDP per capita across municipalities.
        """
        self.qli_gdp_pc = gdp_per_pop
        self.qli_spend_pc = spend_per_pop
        gdp_norm = max(params['QLI_GDP_NORM'], 1e-6)
        economic_driver = (max(0.0, gdp_per_pop) / gdp_norm) ** 0.5
        w = params['QLI_TAX_WEIGHT']
        if w:
            spend_norm = max(params['QLI_SPEND_NORM'], 1e-6)
            fiscal_driver = (max(0.0, spend_per_pop) / spend_norm) ** 0.5
            driver = (1 - w) * economic_driver + w * fiscal_driver
        else:
            driver = economic_driver
        logistic_ceiling = max(0.0, 1.0 - self.index / params['QLI_MAX'])
        self.index += params['QLI_GROWTH_RATE'] * driver * logistic_ceiling

    def update_applied_taxes(self, amount, key):
        self.applied_treasure[key] += amount
        self.applied_flow += amount

    def take_applied_flow(self):
        """Public money applied here since the last call, then reset to zero."""
        flow, self.applied_flow = self.applied_flow, 0.0
        return flow

    def update_index(self, value):
        """Kept for backward compatibility; no longer called by the main loop."""
        self.index += value

    def update_index_pop(self, proportion_pop, elasticity):
        """Kept for backward compatibility; no longer called by the main loop."""
        self.index *= proportion_pop ** elasticity

    def __repr__(self):
        return '%s \n QLI: %.2f, \t GDP: %.2f, \t Pop: %s, Commute: %.2f' % (self.name, self.index, self.gdp,
                                                                             self.pop, self.total_commute)