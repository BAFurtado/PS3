"""House values: house prices are rents over the FipeZAP 2010 gross rental yield, and builders' cost per m² is
the Sinapi 2010 average cost of the state (input/sinapi_2010.csv, auxiliary/sinapi_2010.py), the normal finish
standard, across qualities by the CUB/m² 2010 ratios of the low and high standards to the normal one (R-1, medians over
states; input/cub_2010.csv, auxiliary/cub_2010.py)"""
from collections import defaultdict

import numpy as np
import pandas as pd

UF = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN',
      25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC',
      43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}


def cub_ratios(project='R-1'):
    """Low / normal and high / normal CUB/m², medians over the states"""
    cub = pd.read_csv('input/cub_2010.csv', sep=';')
    cub = cub[cub.project == project].pivot(index='uf', columns='standard', values='cub')
    return float((cub.B / cub.N).median()), float((cub.A / cub.N).median())


def census_ap_rent(census, regions):
    """{region id: Census 2010 mean rent} with each region's mean shrunk toward its municipality's by empirical
    Bayes, as world.location.regional_vacancy does for vacancy: m = (n m_r + k m_mun) / (n + k), n the sampled renting
    households of the region, k = s2 / tau2, s2 the municipality's pooled within-region variance of rent, tau2 the
    between-region variance of the mean net of its sampling noise. tau2 <= 0 gives the municipal mean, a region
    without sampled renters takes the municipal mean"""
    by_mun = defaultdict(list)
    for r in regions:
        if int(r) in census.index:
            row = census.loc[int(r)]
            n = int(row.sampled) if np.isfinite(row.sampled) else 0
            by_mun[r[:7]].append((r, n, row.mean_rent if n else np.nan, row.sd_rent if n > 1 else np.nan))
    out = {}
    for rows in by_mun.values():
        sampled = [(r, n, m, s) for r, n, m, s in rows if n > 0]
        if not sampled:
            continue
        n_tot = sum(n for _, n, _, _ in sampled)
        mean = sum(n * m for _, n, m, _ in sampled) / n_tot
        spread = [(n - 1, s * s) for _, n, _, s in sampled if n > 1]
        s2 = sum(d * v for d, v in spread) / sum(d for d, _ in spread) if spread else 0.0
        observed = sum(n * (m - mean) ** 2 for _, n, m, _ in sampled) / n_tot
        tau2 = observed - len(sampled) * s2 / n_tot
        out.update({r: mean for r, *_ in rows})
        if len(sampled) > 1 and tau2 > 0:
            k = s2 / tau2
            out.update({r: (n * m + k * mean) / (n + k) for r, n, m, _ in sampled})
    return out


class HouseValues:
    def __init__(self, params):
        self.rent_ratio = params['RENTAL_YIELD'] / 12
        # Rents keep their level: rent = price x rent_ratio = size x quality x index x INITIAL_RENTAL_PRICE
        self.price_scale = params['INITIAL_RENTAL_PRICE'] / self.rent_ratio
        self.kappa = params['REAIS_PER_MONEY_UNIT']
        self.sinapi = pd.read_csv('input/sinapi_2010.csv', sep=';', index_col='uf').sinapi.to_dict()
        low, high = cub_ratios()
        # Quality 1 builds at the low standard, 2 at the normal one (Sinapi), 4 at the high one; 3 halfway
        self.standards = ([1, 2, 4], [low, 1.0, high])
        # Builders' productivity draw, whose mean is the average cost Sinapi measures
        self.mean_productivity = 1 - params['CONSTRUCTION_FIRM_MARKUP_MULTIPLIER'] * params['MARKUP'] / 2

    def cost_per_m2(self, region_id, quality):
        """Building cost per m² in model money, without land"""
        sinapi = self.sinapi[UF[int(region_id[:2])]]
        return sinapi * float(np.interp(quality, *self.standards)) / self.kappa

    def build_cost(self, region_id, size, quality, productivity):
        return size * self.cost_per_m2(region_id, quality) * productivity / self.mean_productivity

    def census_rent_scale(self, houses, regions, level='acp'):
        """{region id: factor} that makes the mean rent of each ACP's rented houses equal the Census 2010 mean rent
        of its renting households (input/rent_AP_2010.csv, R$ a month over REAIS_PER_MONEY_UNIT).
        level 'acp': one factor for every region of the ACP. level 'ap': each region's factor is proportional to its
        Census mean rent (shrunk toward its municipality's, census_ap_rent) over its expected quality (Census shares
        of input/dpp_2010_quali.csv, as the generator draws them) x its index; house sizes share one distribution in
        every region, so size drops out. The ACP's rented houses keep the Census mean rent."""
        census = pd.read_csv('input/rent_AP_2010.csv', sep=';').set_index('AREAP')
        acp_of = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').drop_duplicates('cod_mun').set_index(
            'cod_mun').ACPs.to_dict()
        acp = {r: acp_of.get(int(r[:7])) for r in regions}
        relative = {r: 1.0 for r in regions}
        if level == 'ap':
            rent = census_ap_rent(census, regions)
            shares = pd.read_csv('input/dpp_2010_quali.csv', dtype={'areap': str}).set_index('areap')
            quality = shares[['DPP0', 'DPP1', 'DPP2', 'DPP3', 'DPP4', 'DPP5']] @ np.array([.5, 1, 2, 3, 4, 5])
            relative = {r: rent[r] / (quality[r] * regions[r].index) for r in regions
                        if r in rent and quality.get(r, 0) > 0 and regions[r].index > 0}
        model = defaultdict(list)
        for h in houses:
            if h.rent_data is not None and h.family_id is not None and h.region_id in relative:
                model[acp[h.region_id]].append(float(h.rent_data[0]) * relative[h.region_id])
        data = defaultdict(lambda: [0.0, 0.0])
        for r in regions:
            if int(r) in census.index:
                row = census.loc[int(r)]
                if row.renters > 0 and np.isfinite(row.mean_rent):
                    data[acp[r]][0] += row.renters * row.mean_rent
                    data[acp[r]][1] += row.renters
        factor = {a: data[a][0] / data[a][1] / self.kappa / np.mean(model[a])
                  for a in model if data[a][1] > 0 and np.mean(model[a]) > 0}
        return {r: factor[a] * relative[r] for r, a in acp.items() if a in factor and r in relative}

    def upgrade_cost(self, region_id, size, productivity):
        """Works that take a quality .5 house to quality 1: half the cost of building at quality 1"""
        return .5 * self.build_cost(region_id, size, 1, productivity)
