import pandas as pd

# This is Table 14--Matriz dos coeficientes tecnicos intersetoriais D.Bn 2015 from IBGE
technical_matrix = pd.read_csv('input/technical_matrix.csv')
# This is Table 03--Matriz Oferta e demanda da produção nacional a preço básico - 2015 from IBGE
# NGOs consumption was added to Government consumption
# Data refers only to the final demand part of the table
# StockVariation column is desconsidered (relatively small number and endogenous)
# Numbers refer to percentage of that sector in the total buying demand of that class of consumers (COLUMNS)
final_demand = pd.read_csv('input/final_demand.csv')

def read_technical_matrix(mun_codes): 
    if not isinstance(mun_codes, list):
        mun_codes =[mun_codes,]
    technical_matrix = {mun_code: pd.read_json('input/technical matrix/'+mun_code+'_matrix_io.json') for mun_code in mun_codes}
    sector_names = technical_matrix.index.str.split('_', expand=True)[1].unique()
    n = len(sector_names)-1
    #Splitting the matrix into the 4 region destination and origin
    #Input direction origin->destination:
    #LOCAL->LOCAL, EXTERNAL->LOCAL, LOCAL->EXTERNAL, EXTERNAL->EXTERNAL
    for mun_code in technical_matrix:
        technical_matrix[mun_code]={'local_local':technical_matrix.iloc[:n,:n],
                                    'external_local':technical_matrix.iloc[n+1:,:n],
                                    'local_external':technical_matrix.iloc[:n,n+1:],
                                    'external_external':technical_matrix.iloc[n+1:,n+1:]}
    matrix_list = [
        technical_matrix.iloc[:n,:n], technical_matrix.iloc[n+1:,:n],
        technical_matrix.iloc[:n,n+1:],technical_matrix.iloc[n+1:,:n+1:],]
    for m in matrix_list:
        m.index = sector_names
        m.columns = sector_names
    technical_matrix,loc_ext_matrix,ext_local_matrix,ext_ext_matrix = matrix_list
    return technical_matrix,loc_ext_matrix,ext_local_matrix,ext_ext_matrix
class RegionalMarket:
    """
    The regional market contains interactions between productive sectors such as production functions from the
    input-output matrix, creation of externalities and market balancing.
    """

    # TODO: How to handle transport firms? Include a factor of distance by agent/household
    # TODO Include EXPORTS AND FBCF in the consumption market
    # TODO Check pycg (callgraph)

    def __init__(self, sim):
        self.sim = sim
        self.technical_matrix, self.loc_ext_matrix,self.ext_local_matrix,self.ext_ext_matrix = read_technical_matrix(sim.geo.processing_acps) #TODO: How are firms locations infos stored
        self.final_demand = final_demand.set_index('sector')
        self.if_origin = self.sim.PARAMS["TAX_ON_ORIGIN"]

    def consume(self):
        # Household consumption
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
                self.if_origin
            )
        # TODO. External demand. Use of the right-side of the 2n2n IO matrix (final/intermediate demands)

    #TODO: External final demand
    def government_consumption(self):
        gov_firms = [f for f in self.sim.firms.values() if f.sector == 'Government']
        for firm in gov_firms:
            firm.consume(self.sim)

    def gross_fixed_capital_formation(self):
        pass

    def exports(self):
        pass

    def intermediate_consumption(self, amount, firm):
        return firm.sale(amount, self.sim.regions, self.sim.PARAMS['TAX_CONSUMPTION'], firm.region_id,
                         if_origin=self.sim.PARAMS['TAX_ON_ORIGIN'])


class External:
    """
        Provision of inputs from all other metropolitan areas
    """
    def __init__(self, sim, tax_consumption):
        self.sim = sim
        self.amount_sold = 0
        self.total_quantity = 10e10
        # Taxes paid go back to 0 every month.
        self.taxes_paid = 0
        self.cumulative_taxes_paid = 0
        self.tax_consumption = tax_consumption

    def get_amount_sold(self):
        return self.amount_sold

    def intermediate_consumption(self, amount):
        """Sell max amount of products for a given amount of money"""
        if amount > 0:
            # Sticking to a SINGLE product for firm
            amount_per_product = amount / 1
            # FREIGHT included for external goods
            bought_quantity = amount / (self.sim.avg_prices * (1 + self.sim.PARAMS['PUBLIC_TRANSIT_COST']))
            self.amount_sold += amount_per_product
            self.total_quantity -= bought_quantity
            self.taxes_paid += amount_per_product * self.tax_consumption
            self.cumulative_taxes_paid += self.taxes_paid

    def collect_transfer_consumption_tax(self):
        taxes = self.taxes_paid * self.tax_consumption
        self.taxes_paid = 0
        return taxes


class ForeignSector:
    """
    Handles imports and exports
    """
