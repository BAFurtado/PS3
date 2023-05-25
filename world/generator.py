"""
This is the module that uses input data to generate the artificial agent entities (instances) used in the model.
First, regions - the actual municipalities - are created using shapefile input of real limits and real urban/rural
areas. Then, Agents are created and bundled into families, given population measures. Then, houses and firms are created
and families are allocated to their first houses.
"""
import logging
import math
import uuid

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from agents import Agent, Family, Firm, ConstructionFirm, Region, House, Central
from .firms import FirmData
from .population import pop_age_data
from .shapes import prepare_shapes

logger = logging.getLogger('generator')

# Necessary input Data
prop_urban = pd.read_csv('input/prop_urban_2000_2010.csv', sep=';')


class Generator:
    def __init__(self, sim):
        self.sim = sim
        self.seed = sim.seed
        self.seed_np = sim.seed_np
        self.urban, self.shapes = prepare_shapes(sim.geo)
        self.firm_data = FirmData(self.sim.geo.year)
        self.central = Central('central')
        single_ap_muns = pd.read_csv(f'input/single_aps_{self.sim.geo.year}.csv')
        self.single_ap_muns = single_ap_muns['mun_code'].tolist()
        self.quali = self.load_quali()

    def years_study(self, loc):
        # Qualification 2010 degrees of instruction transformation into years of study
        parameters = {'1': self.seed.choice(['1', '2']),
                      '2': self.seed.choice(['4', '6', '8']),
                      '3': self.seed.choice(['9', '10', '11']),
                      '4': self.seed.choice(['12', '13', '14', '15']),
                      '5': self.seed.choice(['1', '2', '4', '6', '8', '9'])}
        return parameters[loc]

    def gen_id(self):
        """Generate a random id that should avoid collisions"""
        return str(uuid.uuid4())[:12]

    def create_regions(self):
        """Create regions"""
        idhm = pd.read_csv('input/idhm_2000_2010.csv', sep=';')
        idhm = idhm.loc[idhm['year'] == self.sim.geo.year]
        regions = {}
        for item in self.shapes:
            r = Region(item, 1)
            # mun code is always first 7 digits of id whether it's a municipality shape or an AP shape
            mun_code = r.id[:7]
            r.index = idhm[idhm['cod_mun'] == int(mun_code)]['idhm'].iloc[0]
            regions[r.id] = r
        return regions

    def create_all(self, regions):
        """Based on regions and population data, create agents, families, houses, and firms"""
        my_agents = {}
        my_families = {}
        my_houses = {}
        my_firms = {}

        if self.sim.geo.year == 2010:
            avg_num_fam = pd.read_csv('input/average_num_members_families_2010.csv')

        for region_id, region in regions.items():
            logger.info('Generating region {}'.format(region_id))

            regional_agents = self.create_agents(region)
            for agent in regional_agents.keys():
                my_agents[agent] = regional_agents[agent]

            num_agents = len(regional_agents)
            if self.sim.geo.year == 2010:
                try:
                    num_families = int(num_agents /
                                       avg_num_fam[avg_num_fam['AREAP'] == int(region_id)].iloc[0]['avg_num_people'])
                except KeyError:
                    num_families = int(num_agents / self.sim.PARAMS['MEMBERS_PER_FAMILY'])
            else:
                num_families = int(num_agents / self.sim.PARAMS['MEMBERS_PER_FAMILY'])
            num_houses = int(num_families * (1 + self.sim.PARAMS['HOUSE_VACANCY']))
            num_firms = int(self.firm_data.num_emp_t0[int(region.id)] * self.sim.PARAMS['PERCENTAGE_ACTUAL_POP'])

            regional_families = self.create_families(num_families)
            regional_houses = self.create_houses(num_houses, region)
            regional_firms = self.create_firms(num_firms, region)

            regional_agents, regional_families = self.allocate_to_family(regional_agents, regional_families)

            # Allocating only percentage of houses to ownership.
            owners_size = int((1 - self.sim.PARAMS['RENTAL_SHARE']) * len(regional_houses))

            # Do not allocate all houses to families. Some families (parameter) will have to rent
            regional_families.update(self.allocate_to_households(dict(list(regional_families.items())[:owners_size]),
                                                                 dict(list(regional_houses.items())[:owners_size])))

            # Set ownership of remaining houses for random families
            self.randomly_assign_houses(regional_houses.values(), regional_families.values())

            # Check families that still do not rent house.
            # Run the first Rental Market
            renting = [f for f in regional_families.values() if f.house is None]
            to_rent = [h for h in regional_houses.values() if h.family_id is None]
            self.sim.housing.rental.rental_market(renting, self.sim, to_rent)

            # Saving on almighty dictionary of families
            for family in regional_families.keys():
                my_families[family] = regional_families[family]

            for house in regional_houses.keys():
                my_houses[house] = regional_houses[house]

            for firm in regional_firms.keys():
                my_firms[firm] = regional_firms[firm]

            try:
                assert len([h for h in regional_houses.values() if h.owner_id is None]) == 0
            except AssertionError:
                print('Houses without ownership')

        return my_agents, my_houses, my_families, my_firms

    def create_agents(self, region):
        agents = {}
        pops = self.sim.pops
        pop_cols = list(list(pops.values())[0].columns)
        if not self.sim.PARAMS['SIMPLIFY_POP_EVOLUTION']:
            list_of_possible_ages = pop_cols[1:]
        else:
            list_of_possible_ages = [0] + pop_cols[1:]

        loop_age_control = list(list_of_possible_ages)
        loop_age_control.pop(0)

        for age in loop_age_control:
            for gender in ['male', 'female']:
                code = region.id
                pop = pop_age_data(pops[gender], code, age, self.sim.PARAMS['PERCENTAGE_ACTUAL_POP'])
                # To see a histogram of qualification check test:
                qualification = self.qual(code)
                moneys = self.seed_np.randint(1, 34, size=pop)
                months = self.seed_np.randint(1, 13, size=pop)
                ages = self.seed_np.randint(list_of_possible_ages[(list_of_possible_ages.index(age, ) - 1)] + 1, age,
                                            size=pop)
                for i in range(pop):
                    agent_id = self.gen_id()
                    a = Agent(agent_id, gender, ages[i], qualification, moneys[i], months[i])
                    agents[agent_id] = a
        return agents

    def create_random_agents(self, n_agents):
        """Create random agents by sampling the existing
        agent population and creating clones of the sampled agents"""
        new_agents = {}
        sample = self.seed.sample(list(self.sim.agents.values()), n_agents)
        moneys = self.seed_np.randint(1, 34, size=len(sample))
        for i, a in enumerate(sample):
            agent_id = self.gen_id()
            new_agent = Agent(agent_id, a.gender, a.age, a.qualification, moneys[i], a.month)
            new_agents[agent_id] = new_agent
        return new_agents

    def create_families(self, num_families):
        community = {}
        for _ in range(num_families):
            family_id = self.gen_id()
            community[family_id] = Family(family_id)
        return community

    def allocate_to_family(self, agents, families):
        """Allocate agents to families"""
        agents = list(agents.values())
        self.seed.shuffle(agents)
        fams = list(families.values())
        # Separate adults to make sure all families have at least one adult
        adults = [a for a in agents if a.age > 21]
        chd = [a for a in agents if a not in adults]
        # Assume there are more adults than families
        # First, distribute adults as equal as possible
        for i in range(len(adults)):
            if not adults[i].belongs_to_family:
                fams[i % len(fams)].add_agent(adults[i])

        # Allocate children into random families
        for agent in chd:
            family = self.seed.choice(fams)
            if not agent.belongs_to_family:
                family.add_agent(agent)
        return agents, families

    def get_random_points_in_polygon(self, region, number_addresses=1, addresses=None, multiplier=3):
        """ Addresses within the region. Additional details so that address fall in urban areas, given percentage"""
        if addresses is None:
            addresses = list()
        if hasattr(region, 'addresses'):
            minx, miny, maxx, maxy = region.addresses.bounds
            right_df = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[region.addresses])
        else:
            minx, miny, maxx, maxy = region.bounds
            right_df = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[region])
        # Number of points has to be large enough so that will have enough correct addresses.
        x = self.seed_np.uniform(minx, maxx, number_addresses * multiplier)
        y = self.seed_np.uniform(miny, maxy, number_addresses * multiplier)
        data = pd.DataFrame()
        data['points'] = [Point(coord) for coord in zip(x, y)]
        gdf_points = gpd.GeoDataFrame(data, geometry='points', crs='epsg:4326')
        sjoin = gpd.tools.sjoin(gdf_points, right_df, predicate='within', how='left')
        addresses += sjoin.loc[sjoin.index_right >= 0, 'points'].tolist()
        # Check to see if number has been reached
        while len(addresses) < number_addresses:
            addresses += self.get_random_points_in_polygon(region,
                                                           number_addresses=(number_addresses - len(addresses)),
                                                           addresses=addresses,
                                                           multiplier=multiplier * multiplier)
        return addresses

    def create_houses(self, num_houses, region, addresses=None):
        """Create houses for a region"""
        if addresses is None:
            addresses = list()
        neighborhood = {}
        probability_urban = self.prob_urban(region)
        if probability_urban:
            urban_addresses = int(num_houses * probability_urban)
            urban_region = self.urban[region.id[:7]]
            addresses = self.get_random_points_in_polygon(urban_region, number_addresses=urban_addresses)
        rural = int(num_houses * (1 - probability_urban))
        if rural:
            addresses.append(self.get_random_points_in_polygon(region, number_addresses=rural, addresses=addresses))
        # Distribution of house surface highly skilled to the right, given empirical data
        sizes = self.seed_np.lognormal(np.log(143), .5, size=num_houses)
        # Loose estimate of qualities in the universe
        b, c, d = .38, .3, .1
        qualities = self.seed_np.choice([1, 2, 3, 4], p=[1 - (b + c + d), b, c, d], size=num_houses)
        prices = np.multiply(np.multiply(sizes, qualities), region.index)
        for i in range(num_houses):
            size = sizes[i]
            # Price is given by 4 quality levels
            quality = qualities[i]
            price = prices[i]
            house_id = self.gen_id()
            h = House(house_id, addresses[i], size, price, region.id, quality)
            neighborhood[house_id] = h
        return neighborhood

    def prob_urban(self, region):
        # Only using urban/rural distinction for municipalities with one AP
        mun_code = int(region.id[:7])
        if mun_code in self.single_ap_muns:
            probability_urban = prop_urban[prop_urban['cod_mun'] == int(mun_code)][str(self.sim.geo.year)].iloc[0]
        else:
            probability_urban = 0
        return probability_urban

    def allocate_to_households(self, families, households):
        """Allocate houses to families"""
        unclaimed = list(households)
        self.seed.shuffle(unclaimed)
        house_id = None
        while unclaimed:
            for family in families.values():
                if house_id is None:
                    try:
                        house_id = unclaimed.pop(0)
                    except IndexError:
                        break
                house = households[house_id]
                if not house.is_occupied:
                    family.move_in(house)
                    house.owner_id = family.id
                    family.owned_houses.append(house)
                    house_id = None
        assert len(unclaimed) == 0
        return families

    def randomly_assign_houses(self, houses, families):
        families = list(families)
        houses = [h for h in houses if h.owner_id is None]
        for house in houses:
            family = self.seed.choice(families)
            house.owner_id = family.id
            family.owned_houses.append(house)

    def create_firms(self, num_firms, region):
        sector = {}
        num_construction_firms = math.ceil(num_firms * self.sim.PARAMS['PERCENT_CONSTRUCTION_FIRMS'])
        addresses = self.get_random_points_in_polygon(region, number_addresses=num_firms)
        balances = self.seed_np.beta(1.5, 10, size=num_firms) * 10000
        for i in range(num_firms):
            firm_id = self.gen_id()
            if i < num_construction_firms:
                f = ConstructionFirm(firm_id, addresses[i], balances[i], region.id)
            else:
                f = Firm(firm_id, addresses[i], balances[i], region.id)
            sector[f.id] = f
        return sector

    def load_quali(self):
        quali_sum = pd.read_csv(f'input/qualification_APs_{self.sim.geo.year}.csv')
        quali_sum.set_index('code', inplace=True)
        return quali_sum

    def qual(self, cod):
        sel = self.quali > self.seed.random()
        idx = sel.idxmax(1)
        loc = idx.loc[int(cod)]
        if self.sim.geo.year == 2010:
            return int(self.years_study(loc))
        return int(loc)

