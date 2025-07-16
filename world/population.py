import math

import numpy as np
import pandas as pd
import statsmodels.api as sm


def pop_age_data(pop, code, age, percent_pop):
    """Select and return the proportion value of population
    for a given municipality, gender and age"""
    n_pop = pop[pop['code'] == str(code)][age].iloc[0] * percent_pop
    rounded = int(round(n_pop))

    # for small `percent_pop`, sometimes we get 0
    # when it's better to have at least 1 agent
    if rounded == 0 and math.ceil(n_pop) == 1:
        return 1
    return rounded


def load_pops(mun_codes, params, year):
    """Load populations for specified municipal codes."""
    ap_pops = pd.read_csv(f'input/num_people_age_gender_AP_{year}.csv', sep=';')
    # Extract municipality codes (first 7 digits of AREAP)
    ap_mun_codes = set(int(str(code)[:7]) for code in ap_pops['AREAP'].unique())

    pops = {'male': pd.DataFrame(), 'female': pd.DataFrame()}
    fallback_mun_codes = [code for code in mun_codes if code not in ap_mun_codes]

    if fallback_mun_codes:
        for name, gender in [('men', 'male'), ('women', 'female')]:
            pop = pd.read_csv(f'input/pop_{name}_{year}.csv', sep=';')
            pop = pop[pop['cod_mun'].isin(fallback_mun_codes)]  # Only fallback muns
            pop = pop.rename(columns={'cod_mun': 'code'})
            pops[gender] = pop

    for code, group in ap_pops.groupby('AREAP'):
        if not int(str(code)[:7]) in mun_codes:
            continue
        for gender, gender_code in [('male', 1), ('female', 2)]:
            sub_group = group[group.gender == gender_code][['age', 'num_people']].to_records()
            rows = []
            row = [0 for _ in range(101)]
            for idx, age, count in sub_group:
                row[age] = count
            row = [code] + row
            rows.append(row)

            columns = ['code'] + list(range(101))
            df = pd.DataFrame(rows, columns=columns)
            pops[gender] = pd.concat([pops[gender], df], ignore_index=True)

    for pop in pops.values():
        pop['code'] = pop['code'].astype(np.int64).astype(str)

    total_pop = sum(
        round(pop.iloc[:, pop.columns != 'code'].sum(axis=1).sum(0) * params['PERCENTAGE_ACTUAL_POP']) for pop in
        pops.values())

    # dict male female code 6, 12... and int
    return pops, total_pop


class MarriageData:
    def __init__(self):
        self.data = {'male': {}, 'female': {}}

        for gender, key in [('male', 'men'), ('female', 'women')]:
            for row in pd.read_csv('input/marriage_age_{}.csv'.format(key)).itertuples():
                for age in range(row.low, row.high + 1):
                    self.data[gender][age] = row.percentage

    def p_marriage(self, agent):
        # Probabilities in INPUT table have been adapted to allow marriage only of those 21 or older
        return self.data[agent.gender.lower()].get(agent.age, 0)


pop_estimates = pd.read_csv('input/Demografia/4_Pop_Estimatives_Munic'
                            '/pop_total_munic_estimates_cedeplar_2000_2050.csv',
                            dtype={'year': str, 'mun_code': str}).set_index('mun_code')
marriage_data = MarriageData()


def immigration(sim):
    """Adjust population for immigration"""
    year = str(sim.clock.year)

    # Create new agents for immigration
    for mun_code, pop in sim.mun_pops.items():
        estimated_pop = pop_estimates.at[str(mun_code), year]
        estimated_pop *= sim.PARAMS['PERCENTAGE_ACTUAL_POP']
        # Correction of population by total number of people
        n_immigration = max(estimated_pop - pop, 0)
        n_immigration *= 1 / 12
        n_migrants = math.ceil(n_immigration)
        if not n_migrants:
            continue

        # Get exogenous rate of growth, heads of households and ages
        # People demand is a list of lists containing class_range (age) and count of households
        people_demand = sim.heads.exogenous_new_households()

        new_agents, new_families = dict(), dict()
        for each in people_demand:
            # Create new agents [returns dictionaries]
            new_agents.update(sim.generator.create_random_agents(n_migrants, each))
            # Create new families
            # Find out how number of households in the model are diverging from exogenous expectations
            n_families = max(sim.stats.head_rate[each[1]][sim.clock.months] - each[1], 1)
            new_families.update(sim.generator.create_families(n_families))

        # Assign agents to families
        sim.generator.allocate_to_family(new_agents, new_families)

        # Keep track of new agents & families
        families = []
        for f in new_families.values():
            # Not all families might get members, skip those
            if not f.members:
                continue
            f.savings = sum(m.grab_money() for m in f.members.values())
            families.append(f)

        # Some might have tried to buy houses but failed, pass them directly to the rental market
        homeless = [f for f in families if f.house is None]
        sim.housing.rental.rental_market(homeless, sim)

        # Only keep families that have houses
        families = [f for f in families if f.house is not None]
        for f in families:
            sim.families[f.id] = f

        agents = [a for a in new_agents.values() if a.family in families]

        # Has to come after we allocate households so that we know where the agents live
        for a in agents:
            sim.agents[a.id] = a
            sim.update_pop(None, a.region_id)


class HouseholdsHeads:
    def __init__(self, sim):
        self.sim = sim
        self.head = pd.read_csv('input/Demografia/head_exogenous_example.csv')
        self.head['month'] = pd.to_datetime(self.head['month'])
        self.head['count'] = self.head['count'] * self.sim.PARAMS['PERCENTAGE_ACTUAL_POP']
        self.head['count'] = self.head['count'].round().astype(int)
        self.head = self.head.set_index('month')

    def exogenous_new_households(self):
        # Formation of new households will be exogenous.
        # Compare head_rate existing with exogenous and build the difference
        # Returns a list of lists
        date = self.sim.clock.days.strftime("%Y-%m-%d")
        return self.head[['class_range', 'count']].loc[date].values.tolist()


def marriage(sim):
    """Adjust families for marriages"""
    to_marry = []
    for agent in sim.agents.values():
        if sim.seed_np.rand() < sim.PARAMS['MARRIAGE_CHECK_PROBABILITY']:
            # Compute probability that this agent will marry
            # NOTE we don't consider whether they are already married
            if sim.seed_np.rand() < agent.p_marriage:
                to_marry.append(agent)

    # Marry individuals.
    # NOTE individuals are paired randomly
    sim.seed_np.shuffle(to_marry)
    to_marry = iter(to_marry)
    for a, b in zip(to_marry, to_marry):
        if a.family.id != b.family.id:
            # Characterizing family
            # If both families have other adults, the ones getting married leave family and make a new one
            a_to_move_out = len([m for m in a.family.members.values() if m.age >= 21]) >= 2
            b_to_move_out = len([m for m in b.family.members.values() if m.age >= 21]) >= 2
            if a_to_move_out and b_to_move_out:
                new_family = list(sim.generator.create_families(1).values())[0]
                old_a = a.family
                old_b = b.family
                a.family.remove_agent(a)
                b.family.remove_agent(b)
                new_family.add_agent(a)
                new_family.add_agent(b)
                new_family.relatives.add(a.id)
                new_family.relatives.add(b.id)
                sim.housing.rental.rental_market([new_family], sim)

                # Reverse marriage if they can't find a house
                if new_family.house is None:
                    old_a.add_agent(a)
                    old_b.add_agent(b)
                else:
                    sim.families[new_family.id] = new_family
                    a_region_id = a.family.region_id
                    b_region_id = b.family.region_id
                    sim.update_pop(a_region_id, new_family.house.region_id)
                    sim.update_pop(b_region_id, new_family.house.region_id)

            elif b_to_move_out:
                b.family.remove_agent(b)
                a.family.add_agent(b)
            elif a_to_move_out:
                a.family.remove_agent(a)
                b.family.add_agent(a)
            else:
                # Else adult B and children (if any) move in with A.
                # Transfer ownership, if any
                # Copy list, so we don't modify the list as we iterate
                houses = [h for h in b.family.owned_houses]
                for house in houses:
                    b.family.owned_houses.remove(house)
                    a.family.owned_houses.append(house)
                    house.owner_id = a.family.id

                old_r_id = b.region_id
                id = b.family.id
                b.family.house.empty()

                # Move out of existing rental
                for house in sim.houses.values():
                    if house.family_id == id:
                        house.family_id = None
                        house.rent_data = None

                for each in b.family.members.values():
                    a.family.add_agent(each)
                    sim.update_pop(b.region_id, a.family.region_id)

                savings = b.family.grab_savings(sim.central, sim.clock.year, sim.clock.months)
                a.family.update_balance(savings)
                if id in sim.central.loans:
                    loans = sim.central.loans.pop(id)
                    sim.central.loans[a.family.id] = loans

                del sim.families[id]
                unassigned_houses = [h for h in sim.houses.values() if h.owner_id == id]
                assert len(unassigned_houses) == 0
