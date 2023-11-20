import pandas as pd


technical_matrix = pd.read_csv("input/technical_matrix.csv")


externalities_matrix = pd.read_csv(
    "input/externalities_matrix.csv", sep=";", header=0, decimal=","
)

market_targets = pd.read_csv(
    "input/pct_demand_supply.csv", sep=";", header=0, decimal=","
)


class RegionalMarket:
    """
    The regional market contains interactions between productive sectors such as production functions from the
    input-output matrix, creation of externalities and market balancing.
    """

    def __init__(self, sim):
        self.technical_matrix = technical_matrix
        self.externalities_matrix = externalities_matrix
        self.market_targets = market_targets
        self.sim = sim

    def consume(self):
        firms = list(self.sim.consumer_firms.values())
        if_origin = self.sim.PARAMS["TAX_ON_ORIGIN"]
        for family in self.sim.families.values():
            family.consume(
                firms,
                self.sim.central,
                self.sim.regions,
                self.sim.PARAMS,
                self.sim.seed,
                self.sim.clock.year,
                self.sim.clock.months,
                if_origin,
            )

    def intermediate_consumption(self, amount, firm):
        firm.sale(amount, self.sim.regions, self.sim.PARAMS['TAX_CONSUMPTION'], firm.region_id,
                  if_origin=self.sim.PARAMS['TAX_ON_ORIGIN'])


class OtherRegions:
    """
    Summary of all other metropolitan areas
    """


class ForeignSector:
    """
    test
    """
