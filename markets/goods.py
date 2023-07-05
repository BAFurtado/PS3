
def consume(sim):
    firms = list(sim.consumer_firms.values())
    origin = sim.PARAMS['TAX_ON_ORIGIN']
    for family in sim.families.values():
        family.consume(firms, sim.central, sim.regions, sim.PARAMS, sim.seed, sim.clock.year, sim.clock.months, origin)
