"""Education level by age.

An agent's level (1 sem instrução e fundamental incompleto, 2 fundamental completo e médio incompleto, 3 médio completo
e superior incompleto, 4 superior completo) is drawn from its weighting area's distribution for everyone aged 10+
(input/education_AP_2010.csv, Sidra 1554, "não determinado" left out) reweighted by its municipality's distribution for
the agent's age group over the one for everyone aged 10+ (input/education_age_2010.csv, Sidra 3572). Areas without data
take their municipality's distribution.
Agents under 25 draw the level of the 25-29 group as the one they will finish with and hold, at each age, the highest
level the school ages allow: fundamental completo at 15, médio completo at 18, superior completo at 22. Newborns draw
from their mother's weighting area. Municipalities without Census population take the run's municipalities together."""
from bisect import bisect_right

import numpy as np
import pandas as pd

FILE = 'input/education_age_2010.csv'
AREAS = 'input/education_AP_2010.csv'
LEVELS = [1, 2, 3, 4]
# Years of study of each level, as the generator assigns them
YEARS = {1: [1, 2], 2: [4, 6, 8], 3: [9, 10, 11], 4: [12, 13, 14, 15]}
# Age at which each level is completed; below 15 only level 1 is held
COMPLETION_AGE = {2: 15, 3: 18, 4: 22}
FINAL_GROUP = 25


class Education:
    def __init__(self, mun_codes, seed):
        self.seed = seed
        areas = pd.read_csv(AREAS, sep=';', dtype={'area': str}).set_index('area')[[str(l) for l in LEVELS]]
        areas = areas[areas.index.str[:7].isin([str(m) for m in mun_codes]) & (areas.sum(axis=1) > 0)]
        self.areas = dict(zip(areas.index, (areas.div(areas.sum(axis=1), axis=0)).to_numpy()))
        table = pd.read_csv(FILE, sep=';')
        table = table[table.cod_mun.isin([int(m) for m in mun_codes])]
        self.groups = sorted(table.age_group.unique())
        pooled = table.groupby(['age_group', 'level'])['pop'].sum().unstack()[LEVELS]
        by_mun = table.groupby(['cod_mun', 'age_group', 'level'])['pop'].sum().unstack()[LEVELS].fillna(0.0)
        self.ratio = {}
        for (mun, group), row in by_mun.iterrows():
            if row.sum() <= 0:
                continue
            all_ages = by_mun.loc[mun].sum()
            self.ratio[(str(mun), group)] = (row / row.sum()).to_numpy() / (all_ages / all_ages.sum()).to_numpy()
        self.municipal = {str(m): (t / t.sum()).to_numpy() for m, t in by_mun.groupby(level=0).sum().iterrows()
                          if t.sum() > 0}
        all_ages = pooled.sum()
        self.base = (all_ages / all_ages.sum()).to_numpy()
        self.pooled = {g: (pooled.loc[g] / pooled.loc[g].sum()).to_numpy() / (all_ages / all_ages.sum()).to_numpy()
                       for g in self.groups}

    def group(self, age):
        return self.groups[max(0, bisect_right(self.groups, max(age, FINAL_GROUP)) - 1)]

    def draw_level(self, area, age):
        group = self.group(age)
        ratio = self.ratio.get((area[:7], group))
        if ratio is None or not np.all(np.isfinite(ratio)):
            ratio = self.pooled[group]
        base = self.areas.get(area, self.municipal.get(area[:7], self.base))
        p = base * np.nan_to_num(ratio)
        if p.sum() <= 0:
            p = np.nan_to_num(self.pooled[group])
        i = int(np.searchsorted(np.cumsum(p / p.sum()), self.seed.random_sample(), side='right'))
        return LEVELS[min(i, len(LEVELS) - 1)]

    def draw(self, area, age):
        """Final years of study and the years held at `age`"""
        target = int(self.seed.choice(YEARS[self.draw_level(area, age)]))
        return target, attained(target, age)


def attained(target, age):
    reached = max([1] + [level for level, a in COMPLETION_AGE.items() if age >= a])
    return min(target, YEARS[reached][-1])
