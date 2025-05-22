import pandas as pd
from collections import defaultdict


class MCMV:
    def __init__(self, sim):
        self.sim = sim
        self.modalidades = pd.read_csv('input/planhab_funds/construcao.csv')
        self.select_regions()
        self.policy_money = defaultdict(float)

    def select_regions(self):
        muns = (mun[:7] for mun in self.sim.regions.values())
        self.modalidades = self.modalidades[self.modalidades['cod_ibge'].isin(muns)]

    def update_policy_money(self, year, modalidade):
        df = self.modalidades
        self.policy_money = defaultdict(float)
        for mun in df['cod_ibge'].unique():
            value = df.loc[
                (df['txt_modalidade'] == modalidade) &
                (df['cod_ibge'] == mun) &
                (df['year'] == year),
                'val_desembolsado'
            ].squeeze()
            self.policy_money[mun] += value / 1000 * self.sim.PARAMS['PERCENTAGE_ACTUAL_POP']
        return self.policy_money




