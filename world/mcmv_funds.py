import pandas as pd
from collections import defaultdict
import conf


class MCMV:
    def __init__(self, sim):
        self.sim = sim
        self.modalidades = pd.read_csv('input/planhab_funds/construcao.csv')
        self.select_regions()
        self.policy_money = defaultdict(float)

    def select_regions(self):
        # TODO: The data base only has 6 digit mun code. Check if that is ok
        muns = [int(str(mun)[:6]) for mun in self.sim.geo.mun_codes]
        self.modalidades = self.modalidades[self.modalidades['cod_ibge'].isin(muns)]

    def update_policy_money(self, year, modalidade):
        df = self.modalidades
        self.policy_money = defaultdict(float)

        # Value is a percentage of the local GDP
        for mun in df['cod_ibge'].unique():
            value = df.loc[
                (df['txt_modalidade'] == modalidade) &
                (df['cod_ibge'] == int(mun)) &
                (df['ano'] == int(year)),
                'val_desembolsado'
            ]
            if value.empty and year >= 2020:
                value = conf.PARAMS['POLICY_MCMV_PERCENTAGE'] 
            elif value.empty:
                value = 0  # TODO: Should we use the regional average?
            else:
                value = float(value.iloc[0])
            self.policy_money[str(mun)] += value * self.sim.stats.last_gdp[mun]
        return self.policy_money




