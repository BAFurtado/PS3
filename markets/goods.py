import pandas as pd
from collections import defaultdict

# This is Table 14--Matriz dos coeficientes tecnicos intersetoriais D.Bn 2015 from IBGE
# technical_matrix = pd.read_csv('input/technical_matrix.csv')
# This is Table 03--Matriz Oferta e demanda da produção nacional a preço básico - 2015 from IBGE
# NGOs consumption was added to Government consumption
# Data refers only to the final demand part of the table
# StockVariation column is desconsidered (relatively small number and endogenous)
# Numbers refer to percentage of that sector in the total buying demand of that class of consumers (COLUMNS)
final_demand = pd.read_csv('input/final_demand.csv')


def read_technical_matrix(mun_codes):
    if not isinstance(mun_codes, list):
        mun_codes = [mun_codes, ]
    tech_matrix = pd.read_json('input/technical_matrices/' + mun_codes[0] + '_matrix_io.json')
    # Using matrix to get sector names
    n = 12
    sector_names = [j.split('_')[1] for j in [i for i in tech_matrix.index][:n]]
    # Splitting the matrix into the 4 region destination and origin
    # Input direction origin->destination:
    # LOCAL->LOCAL, EXTERNAL->LOCAL, LOCAL->EXTERNAL, EXTERNAL->EXTERNAL

    matrix_list = [
        tech_matrix.iloc[:n, :n],
        tech_matrix.iloc[n:, :n],
        tech_matrix.iloc[:n, n:],
        tech_matrix.iloc[n:, n:]
    ]
    # Fixing matrices names
    new_matrix_list = list()
    for m in matrix_list:
        m.index = sector_names
        m.columns = sector_names
        new_matrix_list.append(m)
    local_local, loc_ext_matrix, ext_local_matrix, ext_ext_matrix = new_matrix_list
    return local_local, loc_ext_matrix, ext_local_matrix, ext_ext_matrix


def read_final_demand_matrix(mun_codes):
    if not isinstance(mun_codes, list):
        mun_codes = [mun_codes, ]
    fin_matrix = pd.read_json('input/final_demand/' + mun_codes[0] + '_final_demand.json').T
    # Using matrix to get sector names
    n = int(len(fin_matrix.index) / 2)
    n_d = int(len(fin_matrix.columns) / 2)
    sector_names = [j.split('_')[1] for j in [i for i in fin_matrix.index][:n]]
    demand_names = [j.split('_')[1] for j in [i for i in fin_matrix.columns][:n_d]]
    # Splitting the matrix into the 4 region destination and origin
    # Demand direction origin->destination:
    # LOCAL->LOCAL, EXTERNAL->LOCAL, LOCAL->EXTERNAL, EXTERNAL->EXTERNAL

    matrix_list = [
        fin_matrix.iloc[:n, :n_d],
        fin_matrix.iloc[n:, :n_d],
        fin_matrix.iloc[:n, n_d:],
        fin_matrix.iloc[n:, n_d:]
    ]
    for m in matrix_list:
        m.index = sector_names
    # Calculating the external demand multiplier:
    # ext_demand = multiplier * internal_demand
    external_demand_multiplier = {}
    for sector in sector_names:
        b = sum(matrix_list[2].loc[sector, :])
        external_demand_multiplier[sector] = b / (1 - b)
    return external_demand_multiplier


class RegionalMarket:
    """
    The regional market contains interactions between productive sectors such as production functions from the
    input-output matrix, creation of externalities and market balancing.
    """

    # TODO: *** How to handle transport firms? Include a factor of distance by agent/household (some included external)
    # TODO *** Include FBCF in the consumption market. INCLUDE AN K (kind of technology) DECAYS WITH TIME

    def __init__(self, sim):
        self.sim = sim
        self.technical_matrix, self.loc_ext_matrix, self.ext_local_matrix, self.ext_ext_matrix = read_technical_matrix(
            sim.geo.processing_acps)

        self.if_origin = self.sim.PARAMS["TAX_ON_ORIGIN"]
        self.final_demand = final_demand
        self.final_demand.index = self.technical_matrix.index
        self.external_demand_multiplier = read_final_demand_matrix(sim.geo.processing_acps)
        self.monthly_hh_consumption = defaultdict(float)
        self.monthly_gov_consumption = defaultdict(float)

    def consume(self):
        self.monthly_hh_consumption = defaultdict(float)
        # Household consumption

        # Create sector-wise dictionary to reduce filtering within families
        firms_by_sector = {
            sector: [f for f in self.sim.firms.values() if f.sector == sector and f.get_total_quantity() > 0]
            for sector in self.sim.regional_market.final_demand.index
        }
        for family in self.sim.families.values():
            consumption = family.consume(
                self,
                self.sim.seed,
                self.sim.central,
                self.sim.regions,
                self.sim.PARAMS,
                self.sim.seed_np,
                self.sim.clock.year,
                self.sim.clock.months,
                self.if_origin,
                firms_by_sector
            )
            for key, value in consumption.items():
                self.monthly_hh_consumption[key] += value

    def government_consumption(self):
        self.monthly_gov_consumption = defaultdict(float)
        gov_firms = [f for f in self.sim.firms.values() if f.sector == 'Government']
        for firm in gov_firms:
            consumption = firm.consume(self.sim)
            for key, value in consumption.items():
                self.monthly_gov_consumption[key] += value

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

    def get_external_amount_sold(self):
        return self.amount_sold

    def intermediate_consumption(self, amount, price=1):
        """ Sell max amount of products for a given amount of money """
        if amount > 0:
            # Sticking to a SINGLE product for firm
            amount_per_product = amount / 1
            # FREIGHT included for external goods
            bought_quantity = amount / price
            self.amount_sold += amount_per_product
            self.total_quantity -= bought_quantity
            self.taxes_paid += amount_per_product * self.tax_consumption
            self.cumulative_taxes_paid += self.taxes_paid

    def choose_firms_per_sector(self, firms, seed_np):
        """
        Choose local firms to buy inputs from
        """
        params = self.sim.PARAMS
        chosen_firms, chosen_firm = {}, None

        for sector in self.sim.regional_market.technical_matrix.index:
            n_firms = len([f for f in firms.values() if (f.sector == sector)])
            # TODO. Consider a higher (proportional) number of firms (> 3*) to benefit from external demand?
            market = seed_np.choice(
                [f for f in firms.values() if f.sector == sector],
                size=min(n_firms,
                         3 * int(params['SIZE_MARKET'])),
                replace=False)
            market = [firm for firm in market if firm.get_total_quantity() > 0]
            if market:
                # Choose 10 firms with the cheapest prices
                market.sort(key=lambda firm: firm.prices)
                chosen_firm = market[0: min(10, n_firms)]
            chosen_firms[sector] = chosen_firm
        return chosen_firms

    def final_consumption(self, internal_final_demand, seed_np):
        """Consumes from local firms according to the regionalized SAM"""
        # Selects a subset of firms to buy from playing the role of rest of Brazil demand from simulated region.
        chosen_firms = self.choose_firms_per_sector(self.sim.firms, seed_np)

        for sector in self.sim.regional_market.technical_matrix.index:
            # Sticking to a SINGLE product for firm
            # External demand is a LINEAR FUNCTION of the internal demand
            if chosen_firms[sector]:
                if ((self.sim.regional_market.external_demand_multiplier[sector]) and
                        (internal_final_demand[sector])):
                    amount_per_product = (self.sim.regional_market.external_demand_multiplier[sector] *
                                          internal_final_demand[sector])
                else:
                    continue
                amount_per_firm = amount_per_product / len(chosen_firms[sector])
                # Buys from firms
                for firm in chosen_firms[sector]:
                    firm.sale(amount_per_firm,
                              self.sim.regions,
                              self.sim.PARAMS['TAX_CONSUMPTION'],
                              firm.region_id,
                              if_origin=self.sim.PARAMS['TAX_ON_ORIGIN'],
                              external=True)

    def collect_transfer_consumption_tax(self):
        taxes = self.taxes_paid * self.tax_consumption
        self.taxes_paid = 0
        self.cumulative_taxes_paid += taxes
        return taxes


class ForeignSector:
    """
    Handles imports and exports
    """
