import os

import numpy as np
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
        # The import share of each product for households and government (set_local_shares)
        self.household_import_share = {}
        self.government_import_share = {}

        self.if_origin = self.sim.PARAMS["TAX_ON_ORIGIN"]
        self.final_demand = final_demand.copy()
        self.final_demand.index = self.technical_matrix.index
        # Households pay rent only in the rental market: the Real Estate share (actual and imputed rent) is 0 and the
        # other sectors' shares are rescaled to sum to 1
        household = self.final_demand['HouseholdConsumption'].copy()
        household['RealEstate'] = 0.0
        self.final_demand['HouseholdConsumption'] = household / household.sum()
        self.monthly_hh_consumption = defaultdict(float)
        # Household money this month meant for each sector, served or not
        self.monthly_hh_intended = defaultdict(float)
        # world.own_account.OwnAccountPools
        self.pools = None
        self.monthly_gov_consumption = defaultdict(float)
        # Diagnostic: household money this month that found no firm of the sector with stock (Family.consume)
        self.household_no_stock = 0.0
        # Diagnostic: household refused quantity that the same sector's leftover stock could have covered, right after
        # the household round (household_refused_coverable)
        self.household_refused_coverable = 0.0
        # Diagnostic: household money this month that firms returned for lack of stock, after any retries
        self.household_unserved = 0.0
        # Household money this month spent outside the ACP
        self.household_imports = 0.0
        # Government money this month meant for each sector, fares paid to Transport, and
        # the quantity of each product firms need as inputs (sector order), for the month-1 trade base
        self.monthly_gov_intended = defaultdict(float)
        self.monthly_fares = 0.0
        # Investment money this month meant for each sector, and investment bought
        self.monthly_inv_intended = defaultdict(float)
        self.monthly_investment = 0.0
        self.input_need = np.zeros(len(self.technical_matrix.index))
        # Pre-compute numpy column arrays to avoid pandas.loc overhead in the per-firm hot loop
        self._sector_order = list(self.technical_matrix.index)
        self._tech_np = {s: self.technical_matrix[s].values.copy() for s in self._sector_order}
        self._ext_local_np = {s: self.ext_local_matrix[s].values.copy() for s in self._sector_order}
        self.local_share = None
        self.national_matrix = pd.read_csv('input/technical_matrix.csv').set_index('sector').loc[
            self._sector_order, self._sector_order].astype(float)
        self.set_local_shares(sim.PARAMS['TRADE_POTENTIAL'])

    def set_local_shares(self, shares):
        """Product i is bought locally in share shares[i] by every buyer. Firms split
        their national input coefficients by it; households and government import 1 - shares[i] of their spending."""
        s = pd.Series(shares, dtype=float).reindex(self._sector_order)
        self.local_share = s.to_dict()
        self.technical_matrix = self.national_matrix.mul(s, axis=0)
        self.ext_local_matrix = self.national_matrix.mul(1 - s, axis=0)
        self._tech_np = {k: self.technical_matrix[k].values.copy() for k in self._sector_order}
        self._ext_local_np = {k: self.ext_local_matrix[k].values.copy() for k in self._sector_order}
        self.household_import_share = {k: 1 - v for k, v in self.local_share.items() if v < 1}
        self.government_import_share = dict(self.household_import_share)

    def consume(self):
        self.monthly_hh_consumption = defaultdict(float)
        self.monthly_hh_intended = defaultdict(float)
        self.household_no_stock = 0.0
        self.household_unserved = 0.0
        self.household_imports = 0.0
        self.monthly_fares = 0.0
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
            if len(transport_firms) <= size_market:
                market = transport_firms
            else:
                market = self.sim.seed.sample(transport_firms, size_market)
            firm = min(market, key=lambda f: f.inventory[0].price)
            self.monthly_fares += money
            change = firm.sale(money, self.sim.regions, params['TAX_CONSUMPTION'], region.id, self.if_origin)
            region.treasure['transport'] = change
            self.monthly_hh_consumption['Transport'] += money - change

    def government_consumption(self):
        self.monthly_gov_consumption = defaultdict(float)
        self.monthly_gov_intended = defaultdict(float)
        gov_firms = [f for f in self.sim.firms.values() if f.sector == 'Government']
        for firm in gov_firms:
            consumption = firm.consume(self.sim)
            for key, value in consumption.items():
                self.monthly_gov_consumption[key] += value

    def firm_investment(self):
        """The ACP's investment fund is spent over products with the national FBCF composition (final_demand FBCF), the
        import share of each product bought outside, the rest from the cheapest of a sample of local firms with stock;
        tradables no local firm served are imported. What finds no stock stays in the fund."""
        from agents.firm import import_price
        sim = self.sim
        self.monthly_investment = 0.0
        self.monthly_inv_intended = defaultdict(float)
        money = sim.investment_fund
        if money <= 0:
            return
        params = sim.PARAMS
        shares = self.final_demand['FBCF']
        shares = shares[shares > 0] / shares.sum()
        shortage_sectors = params['TRADABLE_SECTORS']
        freight = import_price(params)
        left = 0.0
        for sector, share in shares.items():
            money_this_sector = money * share
            self.monthly_inv_intended[sector] += money_this_sector
            imported = money_this_sector * self.government_import_share.get(sector, 0.0)
            if imported > 0:
                sim.external.intermediate_consumption(imported, freight)
                money_this_sector -= imported
            pool, share = self.pools.payable(sector)
            if pool is not None and share > 0:
                money_this_sector -= pool.receive(money_this_sector * share, sim.regions, params['TAX_CONSUMPTION'],
                                                  pool.region_id, True)
            sector_firms = [f for f in sim.firms.values() if f.sector == sector]
            market = sim.seed.sample(sector_firms, min(len(sector_firms), int(params['SIZE_MARKET'])))
            market = [f for f in market if f.total_quantity > 0]
            if market:
                firm = min(market, key=lambda f: f.prices)
                change = firm.sale(money_this_sector, sim.regions, params['TAX_CONSUMPTION'], firm.region_id,
                                   params['TAX_ON_ORIGIN'], buyer='investment')
            else:
                change = money_this_sector
            if change > 0 and sector in shortage_sectors:
                sim.external.intermediate_consumption(change, freight)
                change = 0.0
            left += change
        sim.investment_fund = left
        self.monthly_investment = money - left

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
        # their tax that returns to the municipalities and exports (final demand from the rest of Brazil).
        # net_position accumulates exports - net imports: negative is a cumulative deficit, i.e. money that left the ACP
        self.imports_month = 0.0
        self.import_tax_month = 0.0
        self.net_position = 0.0
        self.last_month = {'imports': 0.0, 'exports': 0.0}
        # Months of final_consumption so far
        self.months = 0
        # Export quantity per sector and national GDP index of month 1 (trade_base)
        self.trade_exports = None
        self.trade_base_index = None
        # Month-1 components of the trade base, month-1 permanent income, income paid since
        self.trade_components = None
        self.base_permanent_income = 0.0
        self.rebase_income = []
        self.national_gdp = pd.read_csv('input/national_real_gdp.csv', sep=';').set_index('year')['index']

    def get_external_amount_sold(self):
        return self.amount_sold

    def intermediate_consumption(self, amount, price=1):
        """ Sell max amount of products for a given amount of money """
        if amount > 0:
            # Sticking to a SINGLE product for firm
            amount_per_product = amount / 1
            # Freight included in the price of external goods
            bought_quantity = amount / price
            self.amount_sold += amount_per_product
            self.total_quantity -= bought_quantity
            self.taxes_paid += amount_per_product * self.tax_consumption
            self.cumulative_taxes_paid += self.taxes_paid
            self.imports_month += amount
            self.sim.ledger['imports'] -= amount
            # collect_transfer_consumption_tax returns taxes_paid * tax_consumption
            self.import_tax_month += amount_per_product * self.tax_consumption ** 2

    def stocked_firms_per_sector(self, firms):
        """Every firm of the sector with stock, each with its share of the sector's
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

    def national_index(self, year):
        """National real GDP index (2010 = 1); the last published value after it, the first before it"""
        s = self.national_gdp
        return float(s.loc[min(max(year, s.index.min()), s.index.max())])

    @staticmethod
    def sector_price(firms):
        """Stock-weighted mean price over the stocked firms, or the mean over all firms when none has stock"""
        qty = sum(f.total_quantity for f in firms if f.total_quantity > 0)
        return (sum(f.total_quantity * f.prices for f in firms if f.total_quantity > 0) / qty if qty > 0
                else sum(f.prices for f in firms) / len(firms))

    def expected_investment(self, by_sector):
        """Month 1: the investment the private firms will make a month at full capacity, the investment rate times
        their value added at capacity (national input coefficients) less wages and firm tax"""
        sim = self.sim
        from agents.firm import Firm
        market = sim.regional_market
        input_share = market.national_matrix.sum(axis=0)
        surplus = 0.0
        for sector, firms in by_sector.items():
            if sector == 'Government' or not firms:
                continue
            wage_share = Firm.wage_shares[sector]
            value_added = sum(f.last_capacity for f in firms) * self.sector_price(firms) * (1 - input_share[sector])
            surplus += value_added * (1 - wage_share) * (1 - sim.PARAMS['TAX_FIRM'])
        return sim.investment_rate * surplus

    def trade_base(self):
        """Month 1: per product, local output (staff capacity; with TRADE_BASE_OUTPUT 'market' over 1 - the own-account
        pool's part of a purchase, or plus the pool's output under OWN_ACCOUNT_POOL 'census') and local demand (input need, household spending and fares, government spending,
        money over the sector's price); local share s = TRADE_POTENTIAL x min(output / demand, 1) and exports = output -
        s x demand, Construction and Government s = TRADE_POTENTIAL and no exports. Demand includes the expected
        investment. Sets the market's local shares and returns the table. The month-1 components are kept for
        rebase_trade."""
        market = self.sim.regional_market
        by_sector = defaultdict(list)
        for f in self.sim.firms.values():
            by_sector[f.sector].append(f)
        investment = self.expected_investment(by_sector)
        fbcf = market.final_demand['FBCF'] / market.final_demand['FBCF'].sum()
        with_pools = self.sim.PARAMS.get('TRADE_BASE_OUTPUT', 'firms') == 'market'
        rows = {}
        for k, sector in enumerate(market._sector_order):
            firms = by_sector.get(sector, [])
            output = sum(f.last_capacity for f in firms)
            if with_pools and sector not in ('Construction', 'Government'):
                if market.pools.producing:
                    output += market.pools.output(sector)
                else:
                    output /= 1 - market.pools.payable(sector)[1]
            rows[sector] = {'output': output,
                            'price': self.sector_price(firms) if firms else 1.0,
                            'input_need': market.input_need[k], 'household': market.monthly_hh_intended[sector],
                            'government': market.monthly_gov_intended[sector], 'investment': investment * fbcf[sector],
                            'fares': market.monthly_fares if sector == 'Transport' else 0.0}
        self.trade_components = pd.DataFrame(rows).T
        self.base_permanent_income = sum(f.get_permanent_income() for f in self.sim.families.values())
        self.trade_base_index = self.national_index(self.sim.clock.year)
        return self.apply_trade_base(1.0, 'trade_base.csv')

    def apply_trade_base(self, household_scale, name):
        """Local shares and exports from the kept month-1 components, household spending times household_scale"""
        potential = self.sim.PARAMS['TRADE_POTENTIAL']
        table = self.trade_components.copy()
        for sector, r in table.iterrows():
            money = r.household * household_scale + r.government + r.investment
            if sector == 'Transport':
                money += r.fares * household_scale
            table.loc[sector, 'demand'] = r.input_need + (money / r.price if r.price > 0 else 0.0)
            r = table.loc[sector]
            if sector in ('Construction', 'Government'):
                share, exports = potential[sector], 0.0
            else:
                share = potential[sector] * (min(r.output / r.demand, 1.0) if r.demand > 0 else (1.0 if r.output > 0 else 0.0))
                exports = r.output - share * r.demand
            table.loc[sector, 'local_share'], table.loc[sector, 'exports'] = share, exports
        self.sim.regional_market.set_local_shares(table['local_share'].to_dict())
        self.trade_exports = table['exports'].to_dict()
        output = getattr(self.sim, 'output', None)
        if output is not None:
            columns = ['output', 'demand', 'price', 'local_share', 'exports']
            if household_scale != 1.0:
                table['household_scale'] = household_scale
                columns.append('household_scale')
            table[columns].to_csv(os.path.join(output.path, name), index_label='sector')
        return table

    def rebase_trade(self):
        """At the start of month 3 and 4 records the household income the model paid in months 2
        and 3 (wages, profit shares, social transfers); at month 4 recomputes the trade base with month-1 household
        spending scaled by that income over the month-1 permanent income, output kept at month-1 staff capacity"""
        if self.months not in (3, 4):
            return
        paid = sum(a.wage_paid + a.last_profit_share + a.last_transfer for a in self.sim.agents.values()
                   if a.family is not None)
        self.rebase_income.append(paid)
        if self.months == 4 and self.base_permanent_income > 0:
            self.apply_trade_base(np.mean(self.rebase_income) / self.base_permanent_income, 'trade_base_rebased.csv')

    def export_demand(self):
        """External demand in money per sector: the month-1 export quantity of the trade base (trade_base) times the
        national real GDP index relative to month 1, times (price / P_imp) ** -EXPORTS_PRICE_ELASTICITY, at the
        sector's price. P_imp = 1. The sector's price is the stock-weighted mean over its stocked firms, or the mean
        over all its firms when none has stock."""
        params = self.sim.PARAMS
        demand = {}
        self.months += 1
        if self.trade_exports is None:
            self.trade_base()
        self.rebase_trade()
        growth = self.national_index(self.sim.clock.year) / self.trade_base_index
        by_sector = defaultdict(list)
        for f in self.sim.firms.values():
            by_sector[f.sector].append(f)
        for sector, quantity in self.trade_exports.items():
            firms = by_sector.get(sector)
            if not (firms and quantity > 0):
                continue
            price = self.sector_price(firms)
            if price > 0:
                demand[sector] = quantity * growth * price ** (1 - params['EXPORTS_PRICE_ELASTICITY'])
        return demand

    def final_consumption(self):
        """The rest of Brazil buys the exports (export_demand) from every firm of the sector with stock, in proportion
        to the value of its stock, or from all its firms when none has stock"""
        from agents.firm import Firm
        chosen_firms = self.stocked_firms_per_sector(self.sim.firms)
        demand = self.export_demand()
        self.sim.regional_market.input_need[:] = 0.0
        for sector in demand:
            if not chosen_firms[sector]:
                chosen_firms[sector] = [(f, None) for f in self.sim.firms.values() if f.sector == sector]

        exported = 0.0
        pools = self.sim.regional_market.pools
        for sector, amount in demand.items():
            # Sticking to a SINGLE product for firm
            sold = 0.0
            rest = amount
            # The sector's own-account part
            pool, pshare = pools.payable(sector)
            if pool is not None and pshare > 0:
                sold = pool.receive(rest * pshare, self.sim.regions, self.sim.PARAMS['TAX_CONSUMPTION'], pool.region_id,
                                    self.sim.PARAMS['TAX_ON_ORIGIN'], external=True)
                rest -= sold
            # Buys from firms
            for firm, weight in chosen_firms[sector]:
                amount_per_firm = rest / len(chosen_firms[sector]) if weight is None else rest * weight
                sold += amount_per_firm - firm.sale(amount_per_firm,
                                                    self.sim.regions,
                                                    self.sim.PARAMS['TAX_CONSUMPTION'],
                                                    firm.region_id,
                                                    if_origin=self.sim.PARAMS['TAX_ON_ORIGIN'],
                                                    external=True)
            # The consumption tax stays in the ACP only when it is charged at origin
            self.sim.ledger['exports'] += sold * (1 if self.sim.PARAMS['TAX_ON_ORIGIN'] or Firm.product_tax is not None
                                                  else 1 - self.sim.PARAMS['TAX_CONSUMPTION'])
            exported += sold

        self.net_position += exported - (self.imports_month - self.import_tax_month)
        self.last_month = {'imports': self.imports_month, 'exports': exported}

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