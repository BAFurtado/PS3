import conf
import tempfile
from simulation import Simulation


def check(label, cond):
    res = 'PASS' if cond(sim) else 'FAIL'
    print(res, label)


print('Verifying...')

# Keep it short
conf.RUN['TOTAL_DAYS'] = 1000

path = tempfile.gettempdir()
sim = Simulation(conf.PARAMS, path)
sim.initialize()

N_HOUSES = len(sim.houses)

sim.run()

check('Construction increases housing supply', lambda sim: len(sim.houses) > N_HOUSES)
check('Bank is loaning money', lambda sim: sim.central.n_loans() > 0)
check('No families without a house', lambda sim: len([f for f in sim.families.values() if f.house is None]) == 0)
check('No more than one family living in the same house',
      lambda sim: len([f_i for f_i in sim.families.values() if f_i.house.family_id]) == \
                  len(set([f_i for f_i in sim.families.values() if f_i.house.family_id])))

conf.PARAMS['PERCENT_CONSTRUCTION_FIRMS'] = 0.0
sim = Simulation(conf.PARAMS, path)
sim.initialize()
N_HOUSES = len(sim.houses)
sim.run()
check('No construction firms leads to no new houses', lambda sim: len(sim.houses) == N_HOUSES)
