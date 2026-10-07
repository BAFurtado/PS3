from agents import Agent
from .education import attained
from .population import marriage_data


# Importing official Data from IBGE, 2000-2030
# NOTE: There are different DATA available for each year 2000-2030 for each State

def check_demographics(sim, birthdays, year, mortality_men, mortality_women, fertility):
    """Agent life cycles: update agent ages, deaths, and births"""
    # One random number per agent drawn in bulk — independent draws across individuals,
    # not shared within a cohort (which caused mass-death events for same-age groups).
    total_agents = sum(len(agents) for agents in birthdays.values())
    random_numbers = sim.seed_np.random(size=total_agents)
    r_idx = 0
    male = 'male'
    for age, agents in birthdays.items():
        age = age + 1
        # Always compute rounded_age so fertility fallback can use it even when
        # the direct mortality lookup succeeds (avoids NameError in fertility block).
        rounded_age = ((age + 4) // 5) * 5
        try:
            prob_mort_m = mortality_men[age][str(year)]
            prob_mort_f = mortality_women[age][str(year)]
        except KeyError:
            try:
                # New data only contains probability at 0, 1, 5, 10 and from onwards.
                prob_mort_m = mortality_men[rounded_age][str(year)]
                prob_mort_f = mortality_women[rounded_age][str(year)]
            except KeyError:
                # New data also only goes up to 90
                prob_mort_m = mortality_men[90][str(year)]
                prob_mort_f = mortality_women[90][str(year)]
        p_pregnancy = 0
        if 14 < age < 50:
            try:
                p_pregnancy = fertility[age][str(year)]
            except KeyError:
                p_pregnancy = fertility[rounded_age][str(year)]
        for agent in agents:
            agent.age += 1
            r = random_numbers[r_idx]
            r_idx += 1
            if agent.target is not None:
                agent.qualification = max(agent.qualification, attained(agent.target, age))
            agent.p_marriage = marriage_data.p_marriage(agent)
            if agent.gender == male:
                if r < prob_mort_m:
                    die(sim, agent)
            else:
                if 14 < age < 50:
                    pregnant(sim, agent, p_pregnancy)
                # Mortality procedures
                # Extract specific agent data to calculate mortality 'Female'
                if r < prob_mort_f:
                    die(sim, agent)


def birth(sim, mother=None):
    """Similar to create agent, but just one individual. The child draws its final years of study from its mother's
    weighting area"""
    age = 0
    if mother is not None:
        target, qualification = sim.generator.education.draw(str(mother.region_id), age)
    else:
        target = None
        qualification = int(sim.seed.gammavariate(3, 3))
        qualification = [qualification if qualification < 21 else 20][0]
    # Newborns hold no money: the family carries the child (they used to get 20-40 created from nothing)
    money = 0
    month = sim.seed.randrange(1, 13, 1)
    gender = sim.seed.choice(['male', 'female'])
    sim.total_pop += 1
    child = Agent((sim.total_pop - 1), gender, age, qualification, money, month)
    if target is not None:
        child.target = target
    return child


def pregnant(sim, agent, p_pregnancy):
    """An agent is born"""
    if sim.seed_np.rand() < p_pregnancy:
        child = birth(sim, agent)
        agent.family.add_agent(child)
        sim.agents[child.id] = child
        sim.update_pop(None, child.region_id)


def die(sim, agent):
    """An agent dies"""
    sim.grave.append(agent)
    old_region_id = agent.family.region_id
    if agent.is_employed:
        sim.firms[agent.firm_id].obit(agent)
    # This makes the house vacant if all members of a given family have passed
    if agent.family.num_members == 1:
        # Save houses of empty family
        _id = agent.family.id
        inheritance = [h for h in sim.houses.values() if h.owner_id == _id]
        to_empty = [h for h in sim.houses.values() if h.family_id == _id]
        for each in to_empty:
            each.family_id = None
            each.rent_data = None
        # Make houses vacant
        for h in inheritance:
            h.owner_id = None
            agent.family.owned_houses.remove(h)

        # The wallet too: it holds the wage paid after the last consumption (it used to be lost with the agent, #39)
        savings = agent.family.grab_savings(sim.central, sim.clock.year, sim.clock.months) + agent.money
        agent.money = 0
        # Inheritance and debt are drawn from this list, so its order must be stable.
        relatives = [sim.families[i] for i in sorted(agent.family.relatives)
                     if i in sim.families]

        # Eliminate families with no members
        agent.family.remove_agent(agent)
        del sim.families[_id]
        unassigned_houses = [h for h in sim.houses.values() if h.owner_id == _id]
        assert len(unassigned_houses) == 0

        # Redistribute houses, debt, and savings of empty family
        if relatives:
            # Choose a member to get house/houses and debt, if any
            if inheritance:
                lucky_ones = sim.seed.choices(relatives, k=len(inheritance))
                # Most expensive house last. Will pop for the luckiest, so,
                # who gets the most expensive house, also gets the debt, if any
                inheritance.sort(key=lambda h: h.price, reverse=False)
                debtor = lucky_ones.pop()
                sim.generator.randomly_assign_houses([inheritance.pop()], [debtor])
                # If we still have other houses and other relatives, assign randomly
                if inheritance and lucky_ones:
                    sim.generator.randomly_assign_houses(inheritance, lucky_ones)
                # If we have just more houses, give them all to the survivor
                elif inheritance:
                    sim.generator.randomly_assign_houses(inheritance, [debtor])
            else:
                debtor = sim.seed.choice(relatives)

            # Distribute savings equally
            savings_per_relative = savings / len(relatives)
            for f in relatives:
                f.update_balance(savings_per_relative)

            # Distribute debt
            if _id in sim.central.loans:
                loans = sim.central.loans.pop(_id)
                sim.central.loans[debtor.id] = loans

        else:
            # Assign randomly
            sim.generator.randomly_assign_houses(inheritance, sim.families.values())
            # A vacant estate goes to the municipality (Código Civil art. 1.822); it used to be lost (#39)
            sim.regions[old_region_id].collect_taxes(savings, 'transaction')

            # Delete debt
            if _id in sim.central.loans:
                del sim.central.loans[_id]
    else:
        # The wallet stays with the family (it used to be lost with the agent, #39)
        agent.family.savings += agent.money
        agent.money = 0
        agent.family.remove_agent(agent)

    sim.update_pop(old_region_id, None)
    a_id = agent.id
    del sim.agents[a_id]