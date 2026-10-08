from collections import defaultdict
import math

import numpy as np
import pandas as pd


def pop_age_data(pop, code, age, percent_pop):
    """Select and return the proportion value of population
    for a given municipality, gender and age"""
    match = pop[pop['code'] == str(code)]
    if match.empty:
        # Shapefile uses 13-digit AREAP codes; fallback CSVs use 7-digit municipality codes.
        # Try the municipality prefix so peripheral ACP municipalities are not skipped.
        match = pop[pop['code'] == str(code)[:7]]
    if match.empty:
        return 0
    # AP data has integer age columns; fallback CSVs may still have string columns.
    col = age if age in match.columns else str(age)
    n_pop = match[col].iloc[0] * percent_pop
    rounded = int(round(n_pop))

    # Only round up to 1 when n_pop >= 0.5 (standard rounding threshold).
    # math.ceil(n_pop)==1 fires for any n_pop in (0,1], forcing census cells with
    # 1–5 people (at low sampling rates) all to produce 1 agent — overcounting small cells.
    if rounded == 0 and n_pop >= 0.5:
        return 1
    return rounded


def region_counts(pops, code, percent_pop):
    """Agents per (gender, age) in a region, its total rounded once and the cells filled
    by largest remainder, so no cell is lost to rounding."""
    exact = {}
    for gender in ('male', 'female'):
        match = pops[gender][pops[gender]['code'] == str(code)]
        if match.empty:
            match = pops[gender][pops[gender]['code'] == str(code)[:7]]
        for age in range(101):
            col = age if age in match.columns else str(age)
            exact[(gender, age)] = float(match[col].iloc[0]) * percent_pop if not match.empty else 0.0
    counts = {k: int(np.floor(v)) for k, v in exact.items()}
    left = int(round(sum(exact.values()))) - sum(counts.values())
    for k in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)[:max(left, 0)]:
        counts[k] += 1
    return counts


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


marriage_data = MarriageData()
CENSUS_POPULATION = 'input/census_population_2010_2022.csv'


def census_growth(mun_codes):
    """Yearly growth factor of each municipality between the 2010 and 2022 Censuses; 1 where either is missing."""
    census = pd.read_csv(CENSUS_POPULATION, sep=';', index_col='cod_mun')
    growth = {}
    for code in mun_codes:
        row = census.loc[int(code)] if int(code) in census.index else None
        growth[code] = (row.pop_2022 / row.pop_2010) ** (1 / 12) if row is not None and row.pop_2010 > 0 else 1.0
    return growth


def target_population(sim, mun_code):
    """The municipality's population at the start x its 2010-2022 Census growth since then"""
    years = (sim.clock.days - sim.PARAMS['STARTING_DAY']).days / 365.25
    if mun_code not in sim.pop_start:
        return sim.mun_pops[mun_code]
    return sim.pop_start[mun_code] * sim.pop_growth[mun_code] ** years


def immigration(sim):
    """Adjust population for immigration: each municipality's shortfall is housed, and its excess removed, within it"""
    number_new_families = 0

    for mun_code, pop in list(sim.mun_pops.items()):
        estimated_pop = target_population(sim, mun_code)
        # Correction of population by total number of people
        if estimated_pop > pop:
            # Create new agents for immigration
            n_immigration = max(estimated_pop - pop, 0)
            n_immigration *= 1 / 12
            n_migrants = math.ceil(n_immigration)
            if not n_migrants:
                continue
            if sim.PARAMS['EXOGENOUS_HEAD_RATE']:
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
            else:
                # Follow exogenous number of people
                new_agents = sim.generator.create_random_agents(n_migrants)
                new_families = max(1, int(n_migrants /
                                          sim.geo.avg_num_people[int(mun_code)][str(sim.geo.year)]))
                new_families = sim.generator.create_families(new_families)
            # Assign agents to families
            if new_agents:
                sim.generator.allocate_to_family(new_agents, new_families)

            # Keep track of new agents & families
            families = []
            for f in new_families.values():
                # Not all families might get members, skip those
                if not f.members:
                    continue
                families.append(f)

            # Some might have tried to buy houses but failed, pass them directly to the rental market
            homeless = [f for f in families if f.house is None]
            # Only the vacant houses of the municipality whose shortfall they fill
            vacant = [h for h in sim.houses.values()
                      if h.family_id is None and h.family_owner and h.region_id[:7] == mun_code]
            sim.housing.rental.rental_market(homeless, sim, to_rent=vacant)

            # Only keep families that have houses
            families = [f for f in families if f.house is not None]
            number_new_families += len(families)
            for f in families:
                sim.families[f.id] = f

            agents = [a for a in new_agents.values() if a.family in families]

            # Has to come after we allocate households so that we know where the agents live
            for a in agents:
                sim.agents[a.id] = a
                sim.ledger['immigrants'] += a.money
                sim.update_pop(None, a.region_id)

        elif pop > estimated_pop:
            # Delete families
            on_the_roof = pop - int(estimated_pop)
            # Select agents to be removed among the municipality's residents
            pool = [a for a in sim.agents.values() if a.family.region_id[:7] == mun_code]
            agents_to_remove = list(sim.seed_np.choice(pool, replace=False, size=min(on_the_roof, len(pool))))
            while agents_to_remove:
                terminal = agents_to_remove.pop()
                sim.demographics.die(sim, terminal)
    sim.stats.update_new_families(number_new_families)


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


def census_pairs(sim, to_marry):
    """In shuffled order, each unpaired agent takes a partner from the rest by
    SpouseEducation"""
    from world.family_matching import SpouseEducation
    from world.own_account import level
    if sim.generator.spouses is None:
        sim.generator.spouses = SpouseEducation(sim.geo.processing_acps, sim.generator.seed_np)
    pool = defaultdict(list)
    for a in reversed(to_marry):
        pool[level(a)].append(a)
    paired, pairs = set(), []
    for a in to_marry:
        if id(a) in paired:
            continue
        pool[level(a)].remove(a)
        b = sim.generator.spouses.pick(a, pool)
        if b is None:
            break
        paired.update((id(a), id(b)))
        pairs.append((a, b))
    return pairs


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
    # NOTE individuals are paired by the Census couples' education
    sim.seed_np.shuffle(to_marry)
    pairs = census_pairs(sim, to_marry)

    for a, b in pairs:
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
                new_family.relatives.add(old_a.id)
                new_family.relatives.add(old_b.id)
                sim.housing.rental.rental_market([new_family], sim)

                # Reverse marriage if they can't find a house
                if new_family.house is None:
                    old_a.add_agent(a)
                    old_b.add_agent(b)
                else:
                    sim.families[new_family.id] = new_family
                    sim.update_pop(old_a.region_id, new_family.house.region_id)
                    sim.update_pop(old_b.region_id, new_family.house.region_id)

            elif b_to_move_out:
                old_region_id = b.family.region_id
                b.family.remove_agent(b)
                a.family.add_agent(b)
                sim.update_pop(old_region_id, a.family.region_id)
            elif a_to_move_out:
                old_region_id = a.family.region_id
                a.family.remove_agent(a)
                b.family.add_agent(a)
                sim.update_pop(old_region_id, b.family.region_id)
            else:
                merge_households(sim, a, b)


def merge_households(sim, a, b):
    """Adult B and B's household (children, if any) move in with A; B's houses, savings and loans go to A's household"""
    # Transfer ownership, if any
    # Copy list, so we don't modify the list as we iterate
    houses = [h for h in b.family.owned_houses]
    for house in houses:
        b.family.owned_houses.remove(house)
        a.family.owned_houses.append(house)
        house.owner_id = a.family.id

    # b.family changes as soon as b moves, so B's family is held here (#40: the loop below used to move
    # only b's population, and B's savings and deposits were taken from A and given back to A, while B's
    # were destroyed or orphaned in the bank)
    old_b = b.family
    old_region_id = old_b.region_id
    _id = old_b.id
    old_b.house.empty()

    # Move out of existing rental
    for house in sim.houses.values():
        if house.family_id == _id:
            house.family_id = None
            house.rent_data = None

    for each in list(old_b.members.values()):
        a.family.add_agent(each)
        sim.update_pop(old_region_id, a.family.region_id)

    savings = old_b.grab_savings(sim.central, sim.clock.year, sim.clock.months)
    a.family.update_balance(savings)
    if _id in sim.central.loans:
        loans = sim.central.loans.pop(_id)
        sim.central.loans[a.family.id] = loans

    del sim.families[_id]
    unassigned_houses = [h for h in sim.houses.values() if h.owner_id == _id]
    assert len(unassigned_houses) == 0


class UnionRates:
    """Monthly probabilities of forming a union (adults not in one) and of separating (couples, by the woman's age),
    from the yearly rates of input/union_rates_2010.csv"""

    def __init__(self):
        t = pd.read_csv('input/union_rates_2010.csv', sep=';')
        self.ages = {sex: g.age.values for sex, g in t.groupby('sex')}
        self.monthly = {sex: {k: 1 - (1 - g[k].values) ** (1 / 12) for k in ('formation', 'separation')}
                        for sex, g in t.groupby('sex')}

    def p(self, agent, kind):
        sex = agent.gender.lower()
        if agent.age < self.ages[sex][0]:
            return 0.0
        return self.monthly[sex][kind][np.searchsorted(self.ages[sex], agent.age, side='right') - 1]


def household_income(family):
    """The members' current wages, profit shares and transfers, as the start of a new household's permanent income"""
    family.permanent_income = family.total_wage() + sum(m.last_profit_share + m.last_transfer
                                                        for m in family.members.values())
    family.start_permanent_income()


def settle(sim, family):
    """A new household buys a vacant house for sale it can afford, else rents; False when it finds neither"""
    for_sale = sorted((h for h in sim.housing.for_sale if h.family_id is None), key=lambda h: h.id)
    if for_sale and family.savings > 0:
        family.savings_with_loan = family.savings + family.bank_savings + sim.central.max_loan(family)[0]
        family.loan_rate = 'market'
        sim.housing.negotiating(family, for_sale, sim, sim.stats.vacancy_rate)
    if family.house is None:
        sim.housing.rental.rental_market([family], sim)
    return family.house is not None


def separate(sim, woman):
    """The man leaves the couple's household with half its savings; the woman keeps the children, the house and the
    loans. Undone when he finds no house."""
    man, old = woman.partner, woman.family
    new = list(sim.generator.create_families(1).values())[0]
    old.remove_agent(man)
    new.add_agent(man)
    savings = old.grab_savings(sim.central, sim.clock.year, sim.clock.months)
    new.savings = savings / 2
    old.savings += savings - new.savings
    household_income(new)
    if not settle(sim, new):
        old.savings += new.savings
        new.savings = 0
        new.remove_agent(man)
        old.add_agent(man)
        return
    sim.families[new.id] = new
    new.relatives.add(old.id)
    old.relatives.add(new.id)
    woman.partner = man.partner = None
    sim.update_pop(old.region_id, new.region_id)


def lives_with_adults(agent):
    """Another member of the agent's household is 21 or older"""
    return any(m is not agent and m.age >= 21 for m in agent.family.members.values())


def unite(sim, woman, man):
    """The couple forms a household of its own when both live with other adults, else the one who does moves in with
    the other, else the man's household moves in with the woman's. Undone when a new household finds no house."""
    w_out = lives_with_adults(woman)
    m_out = lives_with_adults(man)
    if w_out and m_out:
        old_w, old_m = woman.family, man.family
        new = list(sim.generator.create_families(1).values())[0]
        old_w.remove_agent(woman)
        old_m.remove_agent(man)
        new.add_agent(woman)
        new.add_agent(man)
        household_income(new)
        if not settle(sim, new):
            new.remove_agent(woman)
            new.remove_agent(man)
            old_w.add_agent(woman)
            old_m.add_agent(man)
            return
        sim.families[new.id] = new
        new.relatives.update((old_w.id, old_m.id))
        sim.update_pop(old_w.region_id, new.region_id)
        sim.update_pop(old_m.region_id, new.region_id)
    elif m_out:
        old_region_id = man.family.region_id
        man.family.remove_agent(man)
        woman.family.add_agent(man)
        sim.update_pop(old_region_id, woman.family.region_id)
    elif w_out:
        old_region_id = woman.family.region_id
        woman.family.remove_agent(woman)
        man.family.add_agent(woman)
        sim.update_pop(old_region_id, man.family.region_id)
    else:
        merge_households(sim, woman, man)
    woman.partner, man.partner = man, woman


def unions(sim):
    """Separations, then unions among the adults not in one, women paired with men by the Census couples' education,
    the nearest in age"""
    from world.family_matching import SpouseEducation
    from world.own_account import level
    if sim.union_rates is None:
        sim.union_rates = UnionRates()
    rates = sim.union_rates
    women = [a for a in sim.agents.values() if a.gender.lower() == 'female']
    for woman in women:
        if woman.partner is not None and sim.seed_np.rand() < rates.p(woman, 'separation'):
            if woman.partner.family is woman.family:
                separate(sim, woman)
            else:
                woman.partner.partner = woman.partner = None

    singles = {'male': [], 'female': []}
    for agent in sim.agents.values():
        if agent.partner is None and sim.seed_np.rand() < rates.p(agent, 'formation'):
            singles[agent.gender.lower()].append(agent)
    if sim.generator.spouses is None:
        sim.generator.spouses = SpouseEducation(sim.geo.processing_acps, sim.generator.seed_np)
    sim.seed_np.shuffle(singles['female'])
    pool = defaultdict(list)
    for man in reversed(singles['male']):
        pool[level(man)].append(man)
    for woman in singles['female']:
        man = sim.generator.spouses.pick_nearest(woman, pool)
        if man is None:
            break
        if man.family is not woman.family:
            unite(sim, woman, man)
