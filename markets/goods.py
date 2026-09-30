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
    """ Returns the four blocks of the regionalized technical matrix, each with rows = selling sector and columns =
        buying sector: local->local, external->local (the ACP's imports), local->external, external->external.
        The files are {buyer: {seller: coefficient}}, so read_json gives rows = seller, first 12 = the ACP.
        For every buying column, local->local + external->local is the national coefficient (IO_rest builds the import
        block as national minus local)."""
    if not isinstance(mun_codes, list):
        mun_codes = [mun_codes, ]
    tech_matrix = pd.read_json('input/technical_matrices/' + mun_codes[0] + '_matrix_io.json')
    # Using matrix to get sector names
    n = 12
    sector_names = [j.split('_')[1] for j in [i for i in tech_matrix.index][:n]]
    blocks = [tech_matrix.iloc[:n, :n], tech_matrix.iloc[n:, :n], tech_matrix.iloc[:n, n:], tech_matrix.iloc[n:, n:]]
    for m in blocks:
        m.index = sector_names
        m.columns = sector_names
    local_local, ext_local, local_ext, ext_ext = [m.astype(float) for m in blocks]
    # Where both sectors have no local wage mass the location quotients are 0/0 and the file holds NaN (8 small ACPs).
    # Nothing is bought locally from a sector that is absent, so the whole national coefficient is imported.
    missing = local_local.isna() | ext_local.isna()
    if missing.values.any():
        national = pd.read_csv('input/technical_matrix.csv').set_index('sector').loc[sector_names, sector_names]
        local_local = local_local.mask(missing, 0.0)
        ext_local = ext_local.mask(missing, national)
    return local_local, ext_local, local_ext.fillna(0.0), ext_ext.fillna(0.0)


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


def household_refused_coverable(firms):
    """Household quantity refused for lack of stock that the leftover stock of the same sector could have covered:
    sum over sectors of min(refused, leftover). Goods firms only, as unmet_household. Diagnostic of matching: a
    household is refused by the firm it picked while other firms of the sector still hold stock."""
    refused = defaultdict(float)
    leftover = defaultdict(float)
    for f in firms:
        if f.sector == 'Construction':
            continue
        leftover[f.sector] += max(0.0, f.inventory[0].quantity)
        if f.demand_by_buyer and 'household' in f.demand_by_buyer:
            refused[f.sector] += f.demand_by_buyer['household'][1]
    return sum(min(r, leftover[s]) for s, r in refused.items())


class RegionalMarket:
    """
    The regional market contains interactions between productive sectors such as production functions from the
    input-output matrix, creation of externalities and market balancing.
    """

    # TODO: *** How to handle transport firms? Include a factor of distance by agent/household (some included external)
    # TODO *** Include FBCF in the consumption market. INCLUDE AN K (kind of technology) DECAYS WITH TIME

    def __init__(self, sim):
        self.sim = sim
        self.technical_matrix, self.ext_local_matrix, self.loc_ext_matrix, self.ext_ext_matrix = read_technical_matrix(
            sim.geo.processing_acps)
        # HOUSEHOLD_IMPORTS: the ACP's import share of each tradable product, from the import block as read (before the
        # old behaviour below replaces it)
        self.household_import_share = {}
        if sim.PARAMS.get('HOUSEHOLD_IMPORTS', False):
            local, imported = self.technical_matrix.sum(axis=1), self.ext_local_matrix.sum(axis=1)
            share = (imported / (local + imported)).fillna(0.0)
            self.household_import_share = {s: float(share[s]) for s in sim.PARAMS['HOUSEHOLD_IMPORT_SECTORS']
                                           if share[s] > 0}
        if not sim.PARAMS.get('IO_IMPORTS', False):
            # Old behaviour: firms read the local->external block as their imports, which is ~0, so every ACP bought
            # only the local share of its inputs
            self.ext_local_matrix = self.loc_ext_matrix

        self.if_origin = self.sim.PARAMS["TAX_ON_ORIGIN"]
        self.final_demand = final_demand
        self.final_demand.index = self.technical_matrix.index
        self.external_demand_multiplier = read_final_demand_matrix(sim.geo.processing_acps)
        self.monthly_hh_consumption = defaultdict(float)
        self.monthly_gov_consumption = defaultdict(float)
        # Diagnostic: household money this month that found no firm of the sector with stock (Family.consume)
        self.household_no_stock = 0.0
        # Diagnostic: household refused quantity that the same sector's leftover stock could have covered, right after
        # the household round (household_refused_coverable)
        self.household_refused_coverable = 0.0
        # Diagnostic: household money this month that firms returned for lack of stock, after any retries
        self.household_unserved = 0.0
        # Household money this month spent outside the ACP (HOUSEHOLD_IMPORTS)
        self.household_imports = 0.0
        # Pre-compute numpy column arrays to avoid pandas.loc overhead in the per-firm hot loop
        self._sector_order = list(self.technical_matrix.index)
        self._tech_np = {s: self.technical_matrix[s].values.copy() for s in self._sector_order}
        self._ext_local_np = {s: self.ext_local_matrix[s].values.copy() for s in self._sector_order}

    def consume(self):
        self.monthly_hh_consumption = defaultdict(float)
        self.household_no_stock = 0.0
        self.household_unserved = 0.0
        self.household_imports = 0.0
        # Household consumption

        # Single pass over firms to group by sector, then filter by inventory availability
        sector_map = defaultdict(list)
        for f in self.sim.firms.values():
            sector_map[f.sector].append(f)
        firms_by_sector = {
            sector: [f for f in firms if f.inventory[0].quantity > 0]
            for sector, firms in sector_map.items()
        }
        seed_np = self.sim.seed_np
        for family in self.sim.families.values():
            consumption = family.consume(
                self,
                self.sim.seed,
                seed_np,
                self.sim.central,
                self.sim.regions,
                self.sim.PARAMS,
                self.sim.clock.year,
                self.sim.clock.months,
                self.if_origin,
                firms_by_sector
            )
            for key, value in consumption.items():
                self.monthly_hh_consumption[key] += value
        self.transport_fares(firms_by_sector.get('Transport'))
        self.household_refused_coverable = household_refused_coverable(self.sim.firms.values())

    def transport_fares(self, transport_firms):
        """Commuting costs (Agent.pay_transport) and the employer transport tax (TAX_TRANSPORT) are collected per
        region in treasure['transport'] and paid here as household purchases from Transport firms, the cheapest of a
        sample, like other household consumption. What finds no stock waits for next month. They used to be wiped at
        month end with the other treasure, destroying the money (#38)."""
        if not transport_firms:
            return
        params = self.sim.PARAMS
        size_market = int(params['SIZE_MARKET'])
        for region in self.sim.regions.values():
            money = region.treasure['transport']
            if money <= 0:
                continue
            market = transport_firms if len(transport_firms) <= size_market else \
                self.sim.seed.sample(transport_firms, size_market)
            firm = min(market, key=lambda f: f.inventory[0].price)
            change = firm.sale(money, self.sim.regions, params['TAX_CONSUMPTION'], region.id, self.if_origin)
            region.treasure['transport'] = change
            self.monthly_hh_consumption['Transport'] += money - change

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
                         if_origin=self.sim.PARAMS['TAX_ON_ORIGIN'], buyer='input')


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
        # External account of the ACP. Monthly flows: imports (inputs bought outside, freight included), the part of
        # their tax that returns to the municipalities, exports (final demand from the rest of Brazil) and recycled
        # demand (EXTERNAL_RECYCLING_SHARE). net_position accumulates exports + recycled - net imports: negative is a
        # cumulative deficit, i.e. money that left the ACP
        self.imports_month = 0.0
        self.import_tax_month = 0.0
        self.recycle_pending = 0.0
        self.net_position = 0.0
        self.last_month = {'imports': 0.0, 'exports': 0.0, 'recycled': 0.0}

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
            self.imports_month += amount
            self.sim.ledger['imports'] -= amount
            # collect_transfer_consumption_tax returns taxes_paid * tax_consumption
            self.import_tax_month += amount_per_product * self.tax_consumption ** 2

    def choose_firms_per_sector(self, firms, seed):
        """
        Choose local firms to buy inputs from
        """
        params = self.sim.PARAMS
        chosen_firms = {}

        for sector in self.sim.regional_market.technical_matrix.index:
            n_firms = len([f for f in firms.values() if (f.sector == sector)])
            market = seed.sample(
                [f for f in firms.values() if f.sector == sector],
                min(n_firms, 3 * int(params['SIZE_MARKET'])))
            market = [firm for firm in market if firm.total_quantity > 0]
            # Choose 10 firms with the cheapest prices. None when no firm of the sector has stock, so its exports are
            # not sold by the previous sector's firms (#29)
            market.sort(key=lambda firm: firm.prices)
            chosen_firms[sector] = market[0: min(10, n_firms)] or None
        return chosen_firms

    def stocked_firms_per_sector(self, firms):
        """EXTERNAL_DEMAND_SPREAD = 'stock': every firm of the sector with stock, each with its share of the sector's
        stock value. Every firm is served in full whenever the sector's demand does not exceed that value"""
        stocked = defaultdict(list)
        for f in firms.values():
            if f.total_quantity > 0:
                stocked[f.sector].append(f)
        chosen = {}
        for sector in self.sim.regional_market.technical_matrix.index:
            market = stocked.get(sector)
            if not market:
                chosen[sector] = None
                continue
            values = [f.total_quantity * f.prices for f in market]
            total = sum(values)
            chosen[sector] = [(f, v / total) for f, v in zip(market, values)]
        return chosen

    def final_consumption(self, internal_final_demand, seed):
        """Consumes from local firms according to the regionalized SAM"""
        # Selects a subset of firms to buy from playing the role of rest of Brazil demand from simulated region.
        if self.sim.PARAMS.get('EXTERNAL_DEMAND_SPREAD', 'cheapest') == 'stock':
            chosen_firms = self.stocked_firms_per_sector(self.sim.firms)
        else:
            # Equal split: weight None divides by the number of firms, as the old model did
            chosen_firms = {sector: [(f, None) for f in market] if market else None
                            for sector, market in self.choose_firms_per_sector(self.sim.firms, seed).items()}
        multiplier = self.sim.regional_market.external_demand_multiplier

        # External demand is a LINEAR FUNCTION of the internal demand
        demand = {}
        for sector in self.sim.regional_market.technical_matrix.index:
            if chosen_firms[sector] and multiplier[sector] and internal_final_demand[sector]:
                demand[sector] = multiplier[sector] * internal_final_demand[sector]
        total_demand = sum(demand.values())

        # EXTERNAL_RECYCLING_SHARE: that share of what the ACP paid for imports this month, net of the import tax that
        # returns to the municipalities, comes back as demand for its products, split across sectors like the exports.
        # What its firms cannot serve waits for next month. 1 = balanced trade, 0 = imports leave for good (old model)
        share = self.sim.PARAMS.get('EXTERNAL_RECYCLING_SHARE', 0.0)
        if share > 0:
            self.recycle_pending += share * (self.imports_month - self.import_tax_month)
        recycle = self.recycle_pending if (share > 0 and total_demand > 0) else 0.0

        exported, recycled = 0.0, 0.0
        for sector, amount in demand.items():
            # Sticking to a SINGLE product for firm
            extra = recycle * amount / total_demand if recycle else 0.0
            sold = 0.0
            # Buys from firms
            for firm, weight in chosen_firms[sector]:
                amount_per_firm = (amount + extra) / len(chosen_firms[sector]) if weight is None \
                    else (amount + extra) * weight
                sold += amount_per_firm - firm.sale(amount_per_firm,
                                                    self.sim.regions,
                                                    self.sim.PARAMS['TAX_CONSUMPTION'],
                                                    firm.region_id,
                                                    if_origin=self.sim.PARAMS['TAX_ON_ORIGIN'],
                                                    external=True)
            # The consumption tax stays in the ACP only when it is charged at origin
            self.sim.ledger['exports'] += sold * (1 if self.sim.PARAMS['TAX_ON_ORIGIN']
                                                  else 1 - self.sim.PARAMS['TAX_CONSUMPTION'])
            exported += sold * amount / (amount + extra)
            recycled += sold * extra / (amount + extra)
        self.recycle_pending -= recycled

        self.net_position += exported + recycled - (self.imports_month - self.import_tax_month)
        self.last_month = {'imports': self.imports_month, 'exports': exported, 'recycled': recycled}
        self.imports_month, self.import_tax_month = 0.0, 0.0

    def collect_transfer_consumption_tax(self):
        taxes = self.taxes_paid * self.tax_consumption
        self.taxes_paid = 0
        self.sim.ledger['import_tax'] += taxes
        self.cumulative_taxes_paid += taxes
        return taxes


class ForeignSector:
    """
    Handles imports and exports
    """