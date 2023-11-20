import pandas as pd

# This is Table 14--Matriz dos coeficientes tecnicos intersetoriais D.Bn 2015 from IBGE
technical_matrix = pd.read_csv('input/technical_matrix.csv')
# This is Table 03--Matriz Oferta e demanda da produção nacional a preço básico - 2015 from IBGE
# NGOs consumption was added to Government consumption
# Data refers only to the final demand part of the table
# StockVariation column is desconsidered (relatively small number and endogenous)
# Numbers refer to percentage of that sector in the total buying demand of that class of consumers (COLUMNS)
final_demand = pd.read_csv('input/final_demand.csv')


class RegionalMarket:
    """
    The regional market contains interactions between productive sectors such as production functions from the
    input-output matrix, creation of externalities and market balancing.
    """

    def __init__(self, sim):
        self.technical_matrix = technical_matrix
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
