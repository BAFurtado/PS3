"""FAMILY_MATCHING 'census': partners' education levels follow the Census 2010 couples of the ACP
(input/spouse_education_2010.csv, auxiliary/spouse_education.py)"""
import pandas as pd

from world.own_account import LEVELS, level


class SpouseEducation:
    def __init__(self, acps, seed_np):
        table = pd.read_csv('input/spouse_education_2010.csv', sep=';')
        acp = acps[0] if len(acps) == 1 and acps[0] in set(table.acp) else 'BRASIL'
        table = table[table.acp == acp]
        self.p = {}
        for head in LEVELS:
            t = table[table.head_level == head].set_index('spouse_level').share.reindex(LEVELS, fill_value=0.0)
            self.p[head] = t.values / t.values.sum()
        self.seed_np = seed_np

    def pick(self, partner, pool):
        """Removes from `pool` ({level: [agents]}) and returns an agent whose level is drawn from the Census spouses of
        `partner`'s level; the nearest level that still has agents (lower first) if the drawn one has none"""
        drawn = LEVELS[self.seed_np.choice(len(LEVELS), p=self.p[level(partner)])]
        for lv in sorted((lv for lv in LEVELS if pool.get(lv)), key=lambda lv: (abs(lv - drawn), lv)):
            return pool[lv].pop()
        return None
