import pandas as pd


technical_matrix = pd.read_csv(
    "input/technical_matrix.csv", sep=";", header=0, decimal="."
)
market_targets = pd.read_csv(
    "input/pct_demand_supply.csv", sep=";", header=0, decimal="."
)
externalities_matrix = pd.read_csv(
    "input/externalities_matrix.csv", sep=";", header=0, decimal="."
)

# TODO probably send these variables to PARAMS


def consume(sim):
    firms = list(sim.consumer_firms.values())
    origin = sim.PARAMS["TAX_ON_ORIGIN"]
    for family in sim.families.values():
        family.consume(
            firms,
            sim.central,
            sim.regions,
            sim.PARAMS,
            sim.seed,
            sim.clock.year,
            sim.clock.months,
            origin,
        )


class RegionalMarket:
    """
    The regional market contains interactions between productive sectors such as production functions from the
    input-output matrix, creation of externalities and market balancing.
    """

    def __init__(self):
        self.technical_matrix = technical_matrix
        self.market_targets = market_targets
        self.externalities_matrix = externalities_matrix

    def search_goods_market(self, sector_list: list):
        """
        Lists firms by sector, their available supply and prices, and returns the best buying option. It might need
        to look outside the metropolitan area.
        """

    def create_externalities(self, sector: int, money_output: float):
        """
        Based on empirical data, creates externalities according to money output produced by a given activity.
        """

        externalities = self.externalities_matrix

        externalities_list = []

        for row in externalities[sector]:
            externalities_list = money_output * row

        return externalities_list

    def market_balancing(self, market_distribution):
        """
        Based on the MIP sector, buys inputs to produce a given money output of the activity, creates externalities
        and creates a price based on cost.
        """

        target_percentages = self.market_targets

        market_deviation = market_distribution - target_percentages

        deviation_threshold = 0

        if market_deviation > deviation_threshold:
            pass
        else:
            pass

        # TODO check se está muito diferente a empírica da target e se sim fazer algo para trazer de volta a um
        # percentual próximo dos empíricos


class OtherRegions:
    """
    Summary of all other metropolitan areas
    """


class ForeignSector:
    """
    test
    """
