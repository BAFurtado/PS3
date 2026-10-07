"""House values: house prices are rents over the FipeZAP 2010 gross rental yield, and builders' cost per m² is
the Sinapi 2010 average cost of the state (input/sinapi_2010.csv, auxiliary/sinapi_2010.py), the normal finish
standard, across qualities by the CUB/m² 2010 ratios of the low and high standards to the normal one (R-1, medians over
states; input/cub_2010.csv, auxiliary/cub_2010.py)"""
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

    def upgrade_cost(self, region_id, size, productivity):
        """Works that take a quality .5 house to quality 1: half the cost of building at quality 1"""
        return .5 * self.build_cost(region_id, size, 1, productivity)
