"""CAR_OWNERSHIP 'census': private vehicle (car or motorcycle) by ACP and work-income decile, Census 2010
(input/car_ownership_2010.csv, column p_private, auxiliary/car_ownership.py). has_car means a private vehicle.

Every agent aged 16-70 has one when its own uniform draw, fixed for the run, is below the probability of its group:
group 1-10 is the decile of its last wage among the paid workers, group 0 an agent without a job or a wage. Car
status changes only when the agent changes group.
"""
from zlib import crc32

import numpy as np
import pandas as pd

CARS = 'input/car_ownership_2010.csv'


class CarOwnership:
    def __init__(self, acps, seed):
        table = pd.read_csv(CARS, sep=';')
        table = table[table.acp.isin(acps)]
        missing = set(acps) - set(table.acp)
        if missing:
            raise ValueError('%s has no rows for %s' % (CARS, sorted(missing)))
        column = 'p_private' if 'p_private' in table.columns else 'p_car'
        self.p = {acp: g.sort_values('group')[column].to_numpy() for acp, g in table.groupby('acp')}
        muns = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
        muns = muns[muns.ACPs.isin(acps)]
        self.acp_by_mun = dict(zip(muns.cod_mun.astype(str), muns.ACPs))
        self.seed = seed

    def draw(self, agent):
        u = getattr(agent, 'car_draw', None)
        if u is None:
            u = np.random.RandomState([self.seed, crc32(str(agent.id).encode()), 3]).random_sample()
            agent.car_draw = u
        return u

    def group(self, agent, wage_deciles):
        if wage_deciles is None or agent.firm_id is None or not agent.last_wage or agent.last_wage <= 0:
            return 0
        return min(int(np.searchsorted(wage_deciles, agent.last_wage, side='right')), 9) + 1

    def update(self, agents, wage_deciles):
        for agent in agents:
            if agent.is_minor or agent.is_retired or agent.family is None or agent.family.house is None:
                continue
            acp = self.acp_by_mun.get(str(agent.family.house.region_id)[:7])
            if acp is None:
                continue
            agent.has_car = self.draw(agent) < self.p[acp][self.group(agent, wage_deciles)]
