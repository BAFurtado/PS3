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
        self.technical_matrix = technical_matrix.set_index('sector')
        self.final_demand = final_demand.set_index('sector')
        self.sim = sim

    def consumption(self):
        # Household consumption
        self.consume()

    def consume(self):
        # TODO How to handle transport firms? Include a factor of distance by agent/household
        # TODO Include EXPORTS AND FBCF in the consumption market
        # TODO Use gov budget_proportion to assign wages
        if_origin = self.sim.PARAMS["TAX_ON_ORIGIN"]
        for family in self.sim.families.values():
            family.consume(
                self,
                self.sim.firms,
                self.sim.central,
                self.sim.regions,
                self.sim.PARAMS,
                self.sim.seed,
                self.sim.clock.year,
                self.sim.clock.months,
                if_origin,
            )

    def government_consumption(self):
        gov_firms = [f for f in self.sim.firms.values() if f.sector == 'Government']
        for firm in gov_firms:
            firm.consume(self.sim)

    def gross_fixed_capital_formation(self):
        pass

    def exports(self):
        pass

    def intermediate_consumption(self, amount, firm):
        firm.sale(amount, self.sim.regions, self.sim.PARAMS['TAX_CONSUMPTION'], firm.region_id,
                  if_origin=self.sim.PARAMS['TAX_ON_ORIGIN'])


class OtherRegions:
    """
    Summary of all other metropolitan areas
    """


class ForeignSector:
    """
    Handles imports and exports
    """
