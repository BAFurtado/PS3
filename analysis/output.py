import json
import os
from collections import defaultdict

import conf

AGENTS_PATH = 'StoragedAgents'
if not os.path.exists(AGENTS_PATH):
    os.mkdir(AGENTS_PATH)

# These are the params which specifically affect agent generation.
# We check when these change so we know to re-generate the agent population.
GENERATOR_PARAMS = [
    'MEMBERS_PER_FAMILY',
    'HOUSE_VACANCY',
    'SIMPLIFY_POP_EVOLUTION',
    'PERCENTAGE_ACTUAL_POP',
    'PERC_SUPPLY_SIZE_N_LICENSES_PER_REGION',
    'STARTING_DAY'
]

OUTPUT_DATA_SPEC = {
    'stats': {
        'avg': {
            'groupings': ['month'],
            'columns': 'ALL'
        },
        'columns': ['month',
                    'pop',
                    'price_index',
                    'gdp_index',
                    'gdp_growth',
                    'unemployment',
                    'median_workers',
                    'families_median_wealth',
                    'families_wages_received',
                    'families_commuting',
                    'families_savings',
                    'families_helped',
                    'amount_subsidised',
                    'firms_profit',
                    'firms_median_stock',
                    'firms_avg_eco_eff',
                    'firms_median_wage_paid',
                    'firms_median_innovation_investment',
                    'emissions',
                    'gini_index',
                    'average_utility',
                    'pct_zero_consumption',
                    'rent_default',
                    'inflation',
                    'average_qli',
                    'house_vacancy',
                    'house_price',
                    'house_rent',
                    'affordable',
                    'p_delinquent',
                    'equally',
                    'locally',
                    'fpm',
                    'bank',
                    'emissions_fund',
                    'ext_amount_sold',
                    'affordability_median'
                    ]
    },
    'families': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': ['house_price', 'house_rent', 'total_wage', 'savings', 'num_members']
        },
        'columns': ['month', 'id', 'mun_id', 'house_price', 'house_rent',
                    'total_wage', 'savings', 'num_members']
    },
    'banks': {
        'avg': {
            'groupings': ['month'],
            'columns': 'ALL'
        },
        'columns': ['month', 'balance', 'active_loans', 'mortgage_rate', 'p_delinquent_loans',
                    'mean_loan_age', 'min_loan', 'max_loan', 'mean_loan']
    },
    'houses': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': ['price', 'on_market']
        },
        'columns': ['month', 'id', 'x', 'y', 'size', 'price', 'rent', 'quality', 'qli',
                    'on_market', 'family_id', 'region_id', 'mun_id']
    },
    'firms': {
        'avg': {
            'groupings': ['month', 'firm_id'],
            'columns': ['total_balance$', 'number_employees',
                        'stocks', 'amount_produced', 'price', 'amount_sold',
                        'revenue', 'profit', 'wages_paid']
        },
        'columns': ['month', 'firm_id', 'region_id', 'mun_id',
                    'long', 'lat', 'total_balance$', 'number_employees',
                    'stocks', 'amount_produced', 'price', 'amount_sold',
                    'revenue', 'profit', 'wages_paid', 'input_cost', 
                    'emissions','eco_eff','innov_investment','sector']
    },
    'construction': {
        'avg': {
            'groupings': ['month', 'firm_id'],
            'columns': ['total_balance$', 'number_employees',
                        'stocks', 'amount_produced', 'price', 'amount_sold',
                        'revenue', 'profit', 'wages_paid']
        },
        'columns': ['month', 'firm_id', 'region_id', 'mun_id',
                    'long', 'lat', 'total_balance$', 'number_employees',
                    'stocks', 'amount_produced', 'price', 'amount_sold',
                    'revenue', 'profit', 'wages_paid']
    },
    'regional': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': 'ALL'
        },
        'columns': ['month', 'mun_id', 'commuting', 'pop', 'gdp_region',
                    'regional_gini', 'regional_house_values', 'regional_unemployment',
                    'qli_index', 'gdp_percapita', 'treasure', 'equally', 'locally', 'fpm',
                    'licenses']
    }
}


class Output:
    """Manages simulation outputs"""

    def __init__(self, sim, output_path):
        files = ['stats', 'regional', 'time', 'firms', 'banks',
                 'houses', 'agents', 'families', 'grave', 'construction']

        self.sim = sim
        self.times = []
        self.path = output_path
        self.transit_path = os.path.join(self.path, 'transit')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(self.transit_path)

        for p in files:
            path = os.path.join(self.path, '{}.csv'.format(p))
            setattr(self, '{}_path'.format(p), path)

            # reset files for each run
            if os.path.exists(path):
                os.remove(path)

        self.save_name = '{}/{}_states_{}_acps_{}'.format(
            AGENTS_PATH,
            '_'.join([str(self.sim.PARAMS[name]) for name in GENERATOR_PARAMS]),
            '_'.join(sim.geo.states_on_process),
            '_'.join(sim.geo.processing_acps_codes))

    def save_stats_report(self, sim, bank_taxes):
        # Banks
        bank = sim.central
        active = bank.active_loans()
        n_active = len(active)
        pop = len(sim.agents)
        p_delinquent = len(bank.delinquent_loans()) / n_active if n_active else 0

        firm_results = sim.stats.calculate_firms_metrics(sim.firms)
        price_index, inflation = sim.stats.update_price(sim.firms)
        gdp_index, gdp_growth = sim.stats.calculate_gdp_and_eco_efficiency(sim.firms, sim.regions)
        unemployment = sim.stats.update_unemployment(sim.agents.values(), True, True)

        families_results = sim.stats.calculate_families_metrics(sim.families)
        commuting = sim.stats.update_commuting(sim.families.values())

        average_qli = sim.stats.average_qli(sim.regions)

        house_results = sim.stats.calculate_house_metrics(sim.houses)

        mun_applied_treasure = defaultdict(int)
        mun_applied_treasure['bank'] = bank_taxes
        families_helped = sim.funds.families_subsided
        amount_subsided = sim.funds.money_applied_policy
        # Reset for monthly (not cumulative) statistics
        sim.funds.families_subsided, sim.funds.money_applied_policy = 0, 0
        for k in ['equally', 'locally', 'fpm', 'emissions']:
            mun_applied_treasure[k] = sum(r.applied_treasure[k] for r in sim.regions.values())
        # External
        ext_amount_sold = sim.external.get_external_amount_sold()

        report = f"{sim.clock.days};" \
                 f"{pop:d};" \
                 f"{price_index:.2f};" \
                 f"{gdp_index:.2f};" \
                 f"{gdp_growth:.2f};" \
                 f"{unemployment:.2f};" \
                 f"{firm_results['workers']:.2f};" \
                 f"{families_results['median_wealth']:.2f};" \
                 f"{families_results['median_wages']:.2f};" \
                 f"{commuting:.3f};" \
                 f"{families_results['total_savings']:.2f};" \
                 f"{families_helped:.0f};" \
                 f"{amount_subsided:.3f};" \
                 f"{firm_results['aggregate_profits']:.2f};" \
                 f"{firm_results['median_stock']:.2f};" \
                 f"{firm_results['eco_efficiency']:.2f};" \
                 f"{firm_results['median_wages']:.2f};" \
                 f"{firm_results['innovation_investment']:.2f};" \
                 f"{firm_results['emissions']:.2f};" \
                 f"{families_results['gini']:.3f};" \
                 f"{families_results['avg_utility']:.2f};" \
                 f"{families_results['zero_consumption_ratio']:.2f};" \
                 f"{families_results['rent_default_ratio']:.4f};" \
                 f"{inflation:.4f};" \
                 f"{average_qli:.3f};" \
                 f"{house_results['vacancy_rate']:.2f};" \
                 f"{house_results['average_house_price']:.2f};" \
                 f"{house_results['average_rent_price']:.2f};" \
                 f"{families_results['affordability_ratio']:.2f};" \
                 f"{p_delinquent:.4f};" \
                 f"{mun_applied_treasure['equally']:.4f};" \
                 f"{mun_applied_treasure['locally']:.4f};" \
                 f"{mun_applied_treasure['fpm']:.4f};" \
                 f"{mun_applied_treasure['bank']:.4f};" \
                 f"{mun_applied_treasure['emissions']:.4f};" \
                 f"{ext_amount_sold:.2f};" \
                 f"{families_results['median_affordability']:.2f}\n"

        with open(self.stats_path, 'a') as f:
            f.write(report)

    def save_regional_report(self, sim):
        reports = []
        agents_by_mun = defaultdict(list)
        families_by_mun = defaultdict(list)
        for agent in sim.agents.values():
            mun_id = agent.region_id[:7]
            agents_by_mun[mun_id].append(agent)

        for family in sim.families.values():
            # sometimes family.region_id is None?
            if family.region_id:
                mun_id = family.region_id[:7]
                families_by_mun[mun_id].append(family)
            else:
                families_by_mun[family.region_id].append(family)

        # aggregate regions into municipalities,
        # in case they are APs
        municipalities = defaultdict(list)
        for region in sim.regions.values():
            mun_id = region.id[:7]
            municipalities[mun_id].append(region)

        for mun_id, regions in municipalities.items():
            mun_pop = sum(r.pop for r in regions)
            mun_gdp = sum(r.gdp for r in regions)
            mun_agents = agents_by_mun[mun_id]
            mun_families = families_by_mun[mun_id]
            GDP_mun_capita = sim.stats.update_GDP_capita(sim.firms, mun_id, mun_pop)
            commuting = sim.stats.update_commuting(mun_families)
            mun_gini = sim.stats.calculate_regional_gini(mun_families)
            mun_house_values = sim.stats.calculate_avg_regional_house_price(mun_families)
            mun_unemployment = sim.stats.update_unemployment(mun_agents)
            region.total_commute = commuting

            mun_cumulative_treasure = 0
            licenses = 0
            for r in regions:
                mun_cumulative_treasure += sum(r.cumulative_treasure.values())
                licenses += r.licenses

            mun_applied_treasure = defaultdict(int)
            for k in ['equally', 'locally', 'fpm']:
                mun_applied_treasure[k] = sum(r.applied_treasure[k] for r in regions)

            # average QLI of regions
            mun_qli = sum(r.index for r in regions) / len(regions)

            reports.append('%s;%s;%.3f;%d;%.3f;%.4f;%.3f;%.4f;%.5f;%.3f;%.6f;%.6f;%.6f;%.6f;%s'
                           % (sim.clock.days, mun_id, commuting, mun_pop, mun_gdp, mun_gini, mun_house_values,
                              mun_unemployment, mun_qli, GDP_mun_capita, mun_cumulative_treasure,
                              mun_applied_treasure['equally'],
                              mun_applied_treasure['locally'],
                              mun_applied_treasure['fpm'],
                              licenses))

        with open(self.regional_path, 'a') as f:
            f.write('\n' + '\n'.join(reports))

    def save_data(self, sim):
        # firms data is necessary for plots,
        # so always save
        self.save_firms_data(sim)
        self.save_banks_data(sim)

        for type in conf.RUN['SAVE_DATA']:
            # Skip b/c they are saved anyways above
            if type in ['firms', 'banks']: continue
            save_fn = getattr(self, 'save_{}_data'.format(type))
            save_fn(sim)

    def save_firms_data(self, sim):
        with open(self.firms_path, 'a') as f:
            [f.write('%s; %s; %s; %s; %.3f; %.3f; %.3f; %s; %.3f; %.3f; %.3f ; %.3f; %.3f; %.3f; %.3f; '
                     '%.3f;%.3f;%.3f;%.3f; %s \n' %
                     (sim.clock.days, firm.id, firm.region_id, firm.region_id[:7], firm.address.x,
                      firm.address.y, firm.total_balance, firm.num_employees,
                      firm.get_total_quantity(), firm.amount_produced, firm.inventory[0].price,
                      firm.amount_sold, firm.revenue, firm.profit,
                      firm.wages_paid, firm.input_cost, firm.last_emissions, firm.env_efficiency,
                      firm.inno_inv, firm.sector))
             for firm in sim.firms.values()]
        [f.reset_amount_sold() for f in sim.firms.values()]

        with open(self.construction_path, 'a') as f:
            [f.write('%s; %s; %s; %s; %.3f; %.3f; %.3f; %s; %.3f; %.3f; %.3f ; %.3f; %.3f; %.3f; %.3f \n' %
                     (sim.clock.days, firm.id, firm.region_id, firm.region_id[:7], firm.address.x,
                      firm.address.y, firm.total_balance, firm.num_employees,
                      firm.total_quantity, len(firm.houses_built), firm.mean_house_price(),
                      firm.n_houses_sold, firm.revenue, firm.profit,
                      firm.wages_paid))
             for firm in sim.firms.values()
             if firm.sector == 'Construction']

    def save_agents_data(self, sim):
        with open(self.agents_path, 'a') as f:
            [f.write('%s;%s;%s;%.3f;%.3f;%s;%s;%s;%s;%s;%.3f;%s\n' % (sim.clock.days, agent.region_id,
                                                                      agent.gender, agent.address.x,
                                                                      agent.address.y, agent.id, agent.age,
                                                                      agent.qualification, agent.firm_id,
                                                                      agent.family.id, agent.money,
                                                                      agent.distance))
             for agent in sim.agents.values()]

    def save_grave_data(self, sim):
        with open(self.grave_path, 'a') as f:
            [f.write('%s;%s;%s;%s;%s;%d;%d;%d;%s;%s;%.3f;%.3f;%s\n' % (sim.clock.days, agent.region_id,
                                                                       agent.gender,
                                                                       agent.address.x if agent.address else None,
                                                                       agent.address.y if agent.address else None,
                                                                       agent.id, agent.age,
                                                                       agent.qualification, agent.firm_id,
                                                                       agent.family.id if agent.family else None,
                                                                       agent.money, agent.utility,
                                                                       agent.distance))
             for agent in sim.grave]

    def save_house_data(self, sim):
        with open(self.houses_path, 'a') as f:
            [f.write('%s;%s;%f;%f;%.2f;%.2f;%s;%.1f;%.2f;%.2f;%s;%s;%s\n' % (sim.clock.days,
                                                                             house.id,
                                                                             house.address.x,
                                                                             house.address.y,
                                                                             house.size,
                                                                             house.price,
                                                                             house.rent_data[0] if house.rent_data
                                                                             else '',
                                                                             house.quality,
                                                                             sim.regions[house.region_id].index,
                                                                             house.on_market,
                                                                             house.family_id,
                                                                             house.region_id,
                                                                             house.region_id[:7]))
             for house in sim.houses.values()]

    def save_family_data(self, sim):
        with open(self.families_path, 'a') as f:
            [f.write('%s;%s;%s;%s;%s;%.5f;%.2f;%.2f\n' % (sim.clock.days,
                                                          family.id,
                                                          family.region_id[:7],
                                                          family.house.price if family.house else '',
                                                          family.house.rent_data[0] if family.house.rent_data else '',
                                                          family.total_wage(),
                                                          family.savings,
                                                          family.num_members))
             for family in sim.families.values()]

    def save_banks_data(self, sim):
        bank = sim.central
        with open(self.banks_path, 'a') as f:
            active = bank.active_loans()
            n_active = len(active)
            mean_age = sum(l.age for l in active) / n_active if n_active else 0
            p_delinquent = len(bank.delinquent_loans()) / n_active if n_active else 0
            mn, mx, avg = bank.loan_stats()
            f.write(f"{sim.clock.days};{bank.balance:.3f};{n_active:.2f};"
                    f"{bank.mortgage_rate:.6f};"
                    f"{p_delinquent:.3f};{mean_age:.3f};{mn:.3f};{mx:.3f};{avg:.3f}\n")

    def save_transit_data(self, sim, fname):
        region_ids = conf.RUN['LIMIT_SAVED_TRANSIT_REGIONS']
        firms = {}
        for firm in sim.firms.values():
            if region_ids is None or any(firm.region_id.startswith(r_id) for r_id in region_ids):
                firms[firm.id] = (firm.address.x, firm.address.y)

        houses = {}
        for house in sim.houses.values():
            if region_ids is None or any(house.region_id.startswith(r_id) for r_id in region_ids):
                houses[house.id] = (house.address.x, house.address.y)

        agents = {}
        for agent in sim.agents.values():
            if region_ids is None or any(agent.region_id.startswith(r_id) for r_id in region_ids):
                agents[agent.id] = (agent.address.x, agent.address.y, agent.family.house.id, agent.firm_id,
                                    agent.last_wage)

        path = os.path.join(self.transit_path, '{}.json'.format(fname))
        with open(path, 'w') as f:
            json.dump({
                'firms': firms,
                'houses': houses,
                'agents': agents
            }, f,
                indent=4,
                default=str)
