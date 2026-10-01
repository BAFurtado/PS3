"""SOCIAL_TRANSFERS 'data': federal benefits paid to residents from outside the ACP (input/social_transfers_2010.csv,
auxiliary/social_transfers.py), fixed in 2010 R$.

Each month and municipality, with A agents living there and the 2010 beneficiaries per resident b:
- RGPS: the round(b x A) oldest agents receive the municipality's mean benefit;
- BPC: round(b x A) agents without RGPS, those aged 65 or more first, then the poorest families first, receive its
  mean benefit;
- Bolsa Família: the round(b x A) poorest families receive its mean benefit per family, split over the members.
A family's income for the ranking is its employed members' wages and its profit shares this month, per member.
Municipalities without Census population take the run's pooled rates and values."""
from collections import defaultdict

import pandas as pd

FILE = 'input/social_transfers_2010.csv'
# Program: (beneficiaries column, value column)
PROGRAMS = {'rgps': ('rgps_ben', 'rgps_val'), 'bpc': ('bpc_ben', 'bpc_val'), 'pbf': ('pbf_fam', 'pbf_val')}


def program_rates(row, reais_per_unit):
    """{program: (beneficiaries per resident, mean benefit in model money)}"""
    rates = {}
    for program, (ben, val) in PROGRAMS.items():
        rates[program] = (row[ben] / row['pop'], row[val] / row[ben] / reais_per_unit if row[ben] > 0 else 0.0)
    return rates


class SocialTransfers:
    def __init__(self, mun_codes, reais_per_unit):
        table = pd.read_csv(FILE, sep=';')
        table = table[table.cod_mun.isin([int(m) for m in mun_codes])]
        pooled = program_rates(table[table['pop'] > 0].sum(), reais_per_unit)
        self.rates = {str(int(row['cod_mun'])): program_rates(row, reais_per_unit) if row['pop'] > 0 else pooled
                      for _, row in table.iterrows()}
        self.paid = 0.0

    @staticmethod
    def family_income(family):
        members = family.members.values()
        income = sum(m.last_wage for m in members if m.firm_id is not None and m.last_wage)
        income += sum(m.last_profit_share for m in members)
        return income / max(len(family.members), 1)

    def pay(self, sim):
        """Pays this month's benefits into the recipients' wallets, sets each agent's last_transfer and records the
        inflow in the ledger"""
        by_mun = defaultdict(list)
        for agent in sim.agents.values():
            agent.last_transfer = 0.0
            if agent.family is not None and agent.family.region_id is not None:
                by_mun[agent.family.region_id[:7]].append(agent)
        paid = [0.0]

        def give(agent, amount):
            agent.last_transfer += amount
            agent.money += amount
            paid[0] += amount

        for mun, agents in by_mun.items():
            rates = self.rates.get(mun)
            if rates is None:
                continue
            families = {a.family.id: a.family for a in agents}
            income = {fid: self.family_income(f) for fid, f in families.items()}
            rate, amount = rates['rgps']
            n = round(rate * len(agents))
            pensioners = sorted(agents, key=lambda a: (-a.age, str(a.id)))[:n]
            for a in pensioners:
                give(a, amount)
            rate, amount = rates['bpc']
            taken = {a.id for a in pensioners}
            pool = sorted((a for a in agents if a.id not in taken),
                          key=lambda a: (a.age < 65, income[a.family.id], str(a.id)))
            for a in pool[:round(rate * len(agents))]:
                give(a, amount)
            rate, amount = rates['pbf']
            poorest = sorted(families.values(), key=lambda f: (income[f.id], str(f.id)))[:round(rate * len(agents))]
            for f in poorest:
                for m in f.members.values():
                    give(m, amount / len(f.members))
        sim.ledger['social_transfers'] += paid[0]
        self.paid = paid[0]
        return paid[0]
