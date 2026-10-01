"""PARTICIPATION 'census': who is in the labour force.

An agent aged 17-69 is economically active when its own uniform draw is below the Census 2010 share of active people
for its sex, age group and municipality (input/participation_2010.csv). The draw is fixed per agent, so the same
people stay active as they age, enter as the share rises and leave as it falls. Municipalities without Census
population take the share of the run's municipalities together."""
from bisect import bisect_right
from zlib import crc32

import numpy as np
import pandas as pd

FILE = 'input/participation_2010.csv'


class Participation:
    def __init__(self, mun_codes, seed):
        table = pd.read_csv(FILE, sep=';')
        table = table[table.cod_mun.isin([int(m) for m in mun_codes])]
        self.groups = sorted(table.age_group.unique())
        pooled = table.groupby(['gender', 'age_group'])[['pop', 'active']].sum()
        pooled = (pooled.active / pooled['pop']).to_dict()
        self.rates = {}
        for row in table.itertuples():
            rate = row.active / row.pop if row.pop > 0 else pooled[(row.gender, row.age_group)]
            self.rates[(str(row.cod_mun), row.gender, row.age_group)] = rate
        self.seed = seed
        self.draws = {}
        populated = table[table['pop'] > 0]
        self.unemployment = 1 - populated.employed.sum() / populated.active.sum()

    def draw(self, agent):
        if agent.id not in self.draws:
            self.draws[agent.id] = np.random.RandomState([self.seed, crc32(str(agent.id).encode())]).random_sample()
        return self.draws[agent.id]

    def is_active(self, agent):
        if not 16 < agent.age < 70:
            return False
        group = self.groups[bisect_right(self.groups, agent.age) - 1]
        return self.draw(agent) < self.rates[(agent.region_id[:7], agent.gender, group)]
