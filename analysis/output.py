import json
import logging
import os
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import conf

# Files written as CSV (small, used by averaging pipeline)
_CSV_FILES = {'stats', 'regional', 'time', 'head', 'neighbourhood'}
# Files written as Parquet (large per-agent/family/firm data)
_PARQUET_FILES = {'firms', 'banks', 'houses', 'agents', 'families', 'grave', 'construction'}

AGENTS_PATH = 'StoragedAgents'
if not os.path.exists(AGENTS_PATH):
    os.mkdir(AGENTS_PATH)

# These are the params which specifically affect agent generation.
# We check when these change so we know to re-generate the agent population.
GENERATOR_PARAMS = [
    'MEMBERS_PER_FAMILY',
    'HOUSE_VACANCY',
    'PERCENTAGE_ACTUAL_POP',
    'EXPECTED_LICENSES_PER_REGION',
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
                    'price_level',
                    'gdp_level',
                    'gdp_growth_rate',
                    'gdp_change',
                    'unemployment',
                    'firms_median_employment',
                    "firms_total_employment",
                    'families_median_permanent_income',
                    'families_wages_received',
                    'families_commuting',
                    'families_savings',
                    'families_helped',
                    'new_families',
                    'amount_subsidised',
                    'perc_policy_money_spent',
                    'firms_total_profit',
                    'firms_median_profit',
                    'share_firms_positive_profit',
                    'firms_median_stock',
                    'firms_avg_eco_eff',
                    'firms_median_wage_paid',
                    'firms_wage_per_worker',
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
                    'house_quality',
                    'number_domiciles',
                    'affordable',
                    'p_delinquent',
                    'equally',
                    'locally',
                    'fpm',
                    'bank',
                    'emissions_fund',
                    'ext_amount_sold',
                    # Deciles of rent / permanent income among renting families.
                    # Named distinctly from the affordability_decis_* of earlier
                    # batches, which measure a different quantity: the two column
                    # families must never be pooled.
                    # rent_burden_decis_5 == affordability_median by construction.
                    'rent_burden_decis_1',
                    'rent_burden_decis_2',
                    'rent_burden_decis_3',
                    'rent_burden_decis_4',
                    'rent_burden_decis_5',
                    'rent_burden_decis_6',
                    'rent_burden_decis_7',
                    'rent_burden_decis_8',
                    'rent_burden_decis_9',
                    'rent_burden_decis_10',
                    'affordability_median',
                    # Share of renters whose permanent income is <= 0. Their rent
                    # burden is undefined and lands in the top decile, so this is
                    # what makes rent_burden_decis_10 readable. affordability_median
                    # keeps its original definition for comparability with the
                    # submitted density paper.
                    'pct_renters_zero_income',
                    'perc_fgts_used',
                    'perc_sbpe_used',
                    "active_loans",
                    "loan_requested",
                    "loan_approved",
                    "loan_approval_rate",
                    "credit_stock",
                    "bank_balance",
                    "pct_firms_increase_production",
                    # Every rejection path in Central.request_loan has a counter
                    # here, so approved + all denied_* reconciles exactly to
                    # loan_requested. Adding a path without a counter here breaks
                    # that identity; check_smoke_batch.py asserts it.
                    "denied_existing_loan",
                    "denied_invalid_term",
                    "denied_zero_capped_amount",
                    "denied_no_loan_needed",
                    "denied_affordability",
                    "denied_recursos_fgts",
                    "denied_recursos_sbpe",
                    "denied_funding_keyerror",
                    "denied_liquidity_reserve",
                    "denied_bank_limit",
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
                    'emissions', 'eco_eff', 'innov_investment', ' sector']
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
        'columns': ['month',
                    'mun_id',
                    'commuting',
                    'pop',
                    'gdp_region',
                    'regional_gini',
                    'regional_house_values',
                    'regional_unemployment',
                    'qli_index',
                    'gdp_percapita',
                    'treasure',
                    'equally',
                    'locally',
                    'fpm',
                    'licenses',
                    'affordability_ratio',
                    'median_permanent_income',
                    'median_affordability',
                    # MCMV acquisition diagnostics, per municipality-month.
                    # The six mcmv_stop_* columns are mutually exclusive 0/1
                    # indicators, so their average over runs reads as the
                    # probability of each stopping reason. See
                    # world/funds.py:buy_houses_give_to_families.
                    'mcmv_eligible',
                    'mcmv_units_available',
                    'mcmv_units_bought',
                    'mcmv_money_topup',
                    'mcmv_money_start',
                    'mcmv_money_residual',
                    'mcmv_stop_no_eligible',
                    'mcmv_stop_no_units',
                    'mcmv_stop_families_exhausted',
                    'mcmv_stop_budget',
                    'mcmv_stop_indivisible',
                    'mcmv_stop_units_exhausted',
                    # Melhorias, the second OGU programme, on its own pot.
                    # money_start is topup plus the residual carried over from
                    # previous months; money_residual carries into the next one.
                    'melhorias_eligible',
                    'melhorias_upgrades',
                    'melhorias_money_topup',
                    'melhorias_money_start',
                    'melhorias_money_residual']
    },
    'neighbourhood': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': 'ALL'
        },
        'columns': ['month', 'mun_id', 'neigh_id', 'pop', 'neighbourhood_gdp',
                    'neighbourhood_gdp_percapita', 'neighbourhood_commuting', 'neighbourhood_gini']
    }
}


def _legacy_stats_columns():
    """`stats` layout used by batches before 2026-08-01: no
    denied_zero_capped_amount, no pct_renters_zero_income, and the decile block
    carries affordability_decis_* rather than rent_burden_decis_*."""
    dropped = {'denied_zero_capped_amount', 'denied_no_loan_needed',
               'pct_renters_zero_income'}
    cols = [c for c in OUTPUT_DATA_SPEC['stats']['columns'] if c not in dropped]
    return [c.replace('rent_burden_decis_', 'affordability_decis_') for c in cols]


def _legacy_regional_columns():
    """`regional` layout before the MCMV allocation diagnostics were added."""
    return [c for c in OUTPUT_DATA_SPEC['regional']['columns']
            if not c.startswith(('mcmv_', 'melhorias_'))]


def _legacy_regional_columns_single_pot():
    """`regional` layout with the MCMV diagnostics but no per-programme pots: no
    mcmv_money_topup and no melhorias block."""
    return [c for c in OUTPUT_DATA_SPEC['regional']['columns']
            if c != 'mcmv_money_topup' and not c.startswith('melhorias_')]


LEGACY_COLUMNS = {
    'stats': [_legacy_stats_columns()],
    'regional': [_legacy_regional_columns_single_pot(), _legacy_regional_columns()],
}


def columns_for(kind, n_fields):
    """Column names for a headerless output file that has `n_fields` fields.

    stats.csv and regional.csv are written without a header, so anything reading
    a run directory produced by an older version of this file must use the layout
    that matches *that* file rather than the current spec. Passing too many names
    to pandas silently shifts the surplus into the index instead of failing, so
    resolve the width here and raise when nothing fits.
    """
    current = OUTPUT_DATA_SPEC[kind]['columns']
    if len(current) == n_fields:
        return current
    for legacy in LEGACY_COLUMNS.get(kind, []):
        if len(legacy) == n_fields:
            return legacy
    raise ValueError(
        'No known {} layout has {} columns (current spec has {}).'.format(
            kind, n_fields, len(current)))


class Output:
    """Manages simulation outputs"""

    def __init__(self, sim, output_path):
        all_files = ['stats', 'regional', 'time', 'firms', 'banks',
                     'houses', 'agents', 'families', 'grave', 'construction',
                     'head', 'neighbourhood']

        self.sim = sim
        self.times = []
        self.path = output_path
        self.transit_path = os.path.join(self.path, 'transit')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(self.transit_path)

        for p in all_files:
            ext = 'parquet' if p in _PARQUET_FILES else 'csv'
            path = os.path.join(self.path, '{}.{}'.format(p, ext))
            setattr(self, '{}_path'.format(p), path)
            if os.path.exists(path):
                os.remove(path)

        self._pq_writers = {}

        self.save_name = '{}/{}_states_{}_acps_{}'.format(
            AGENTS_PATH,
            '_'.join([str(self.sim.PARAMS[name]) for name in GENERATOR_PARAMS]),
            '_'.join(sim.geo.states_on_process),
            '_'.join(sim.geo.processing_acps_codes))

    def _write_parquet(self, name, path, data_dict):
        table = pa.table(data_dict)
        # Promote integer columns to float64 so the schema is stable across
        # quarterly batches: a numeric column that happens to be all-zero
        # (inferred as int64) in one batch but has fractional values in
        # another would otherwise crash ParquetWriter.write_table with a
        # schema mismatch.
        for i, field in enumerate(table.schema):
            if pa.types.is_integer(field.type):
                table = table.set_column(i, field.name, table.column(i).cast(pa.float64()))
        if name not in self._pq_writers:
            self._pq_writers[name] = pq.ParquetWriter(path, table.schema, compression='snappy')
        self._pq_writers[name].write_table(table)

    def close(self):
        for writer in self._pq_writers.values():
            writer.close()
        self._pq_writers.clear()

    def save_stats_report(self, sim, bank_taxes):
        # Banks
        bank = sim.central
        active = bank.active_loans()
        n_active = len(active)
        pop = len(sim.agents)
        p_delinquent = len(bank.delinquent_loans()) / n_active if n_active else 0

        firm_results = sim.stats.calculate_firms_metrics(sim.firms)
        price_level, inflation = sim.stats.update_price(sim.firms)
        gdp_level, gdp_growth_rate, gdp_change = sim.stats.calculate_gdp_and_eco_efficiency(sim.firms, sim.regions)
        unemployment = sim.stats.update_unemployment(sim.agents.values(), True, True)

        families_results = sim.stats.calculate_families_metrics(sim.families)
        commuting = sim.stats.update_commuting(sim.families.values())

        average_qli = sim.stats.average_qli(sim.regions)

        house_results = sim.stats.calculate_house_metrics(sim.houses)

        mun_applied_treasure = defaultdict(int)
        mun_applied_treasure['bank'] = bank_taxes
        families_helped = sim.funds.families_subsided
        amount_subsidised = sim.funds.money_applied_policy
        perc_policy_money_spent = sim.funds.perc_policy_money_spent
        # Reset for monthly (not cumulative) statistics
        sim.funds.families_subsided, sim.funds.money_applied_policy = 0, 0
        for k in ['equally', 'locally', 'fpm']:
            mun_applied_treasure[k] = sum(r.applied_treasure[k] for r in sim.regions.values())
        emissions_fund = sum(r.treasure['emissions'] for r in sim.regions.values())
        perc_fgts, perc_sbpe = sim.central.funding_usage_month(
            sim.clock.year,
            sim.clock.months,
            sim.regions.values()
        )
        # External
        ext_amount_sold = sim.external.get_external_amount_sold()

        stats_row = {
            "month": sim.clock.days,
            "pop": pop,
            "price_level": price_level,
            "gdp_level": gdp_level,
            "gdp_growth_rate": gdp_growth_rate,
            "gdp_change": gdp_change,
            "unemployment": unemployment,
            "firms_median_employment": firm_results["workers"],
            "firms_total_employment": firm_results["firms_total_employment"],
            "families_median_permanent_income": families_results["median_permanent_income"],
            "families_wages_received": families_results["median_wages"],
            "families_commuting": commuting,
            "families_savings": families_results["total_savings"],
            "families_helped": families_helped,
            'new_families': families_results["new_families"],
            "amount_subsidised": amount_subsidised,
            "perc_policy_money_spent": perc_policy_money_spent,
            "firms_total_profit": firm_results["aggregate_profits"],
            "firms_median_profit": firm_results["median_profit"],
            "share_firms_positive_profit": firm_results["share_firms_positive_profit"],
            "firms_median_stock": firm_results["median_stock"],
            "firms_avg_eco_eff": firm_results["eco_efficiency"],
            "firms_median_wage_paid": firm_results["median_wages"],
            "firms_wage_per_worker": firm_results["median_wage_per_worker"],
            "firms_median_innovation_investment": firm_results["innovation_investment"],
            "emissions": firm_results["emissions"],
            "gini_index": families_results["gini"],
            "average_utility": families_results["avg_utility"],
            "pct_zero_consumption": families_results["zero_consumption_ratio"],
            "rent_default": families_results["rent_default_ratio"],
            "inflation": inflation,
            "average_qli": average_qli,
            "house_vacancy": house_results["vacancy_rate"],
            "house_price": house_results["average_house_price"],
            "house_rent": house_results["average_rent_price"],
            "house_quality": house_results["mean_quality"],
            "number_domiciles": house_results['number_domiciles'],
            "affordable": families_results["affordability_ratio"],
            "p_delinquent": p_delinquent,
            "equally": mun_applied_treasure["equally"],
            "locally": mun_applied_treasure["locally"],
            "fpm": mun_applied_treasure["fpm"],
            "bank": mun_applied_treasure["bank"],
            "emissions_fund": emissions_fund,
            "ext_amount_sold": ext_amount_sold,
            "affordability_median": families_results["median_affordability"],
            "pct_renters_zero_income": families_results["zero_income_renter_share"],
            "perc_fgts_used": perc_fgts,
            "perc_sbpe_used": perc_sbpe,
            "active_loans": n_active,
            "loan_requested": bank.loan_stats["requested"],
            "loan_approved": bank.loan_stats["approved"],
            "loan_approval_rate": bank.loan_stats["approved"] / bank.loan_stats["requested"] if bank.loan_stats[
                "requested"] else 0,
            "credit_stock": bank._outstanding_loans,
            "bank_balance": bank.balance,
            "pct_firms_increase_production": firm_results["pct_increase_production"],
            "denied_existing_loan": bank.loan_stats["denied_existing_loan"],
            "denied_invalid_term": bank.loan_stats["denied_invalid_term"],
            "denied_zero_capped_amount": bank.loan_stats["denied_zero_capped_amount"],
            "denied_no_loan_needed": bank.loan_stats["denied_no_loan_needed"],
            "denied_affordability": bank.loan_stats["denied_affordability"],
            "denied_recursos_fgts": bank.loan_stats["denied_recursos_fgts"],
            "denied_recursos_sbpe": bank.loan_stats["denied_recursos_sbpe"],
            "denied_funding_keyerror": bank.loan_stats["denied_funding_keyerror"],
            "denied_liquidity_reserve": bank.loan_stats["denied_liquidity_reserve"],
            "denied_bank_limit": bank.loan_stats["denied_bank_limit"],
        }

        for i, v in enumerate(families_results["rent_burden_decis"], start=1):
            stats_row[f"rent_burden_decis_{i}"] = v

        columns = OUTPUT_DATA_SPEC["stats"]["columns"]
        row = ";".join(str(stats_row[c]) for c in columns) + "\n"

        with open(self.stats_path, "a") as f:
            f.write(row)

        logger = (logging.getLogger('bank'))
        #logger.info(dict(sim.central.loan_stats))

    def save_regional_report(self, sim):
        reports = []
        agents_by_mun = defaultdict(list)
        families_by_mun = defaultdict(list)
        for agent in sim.agents.values():
            mun_id = agent.region_id[:7]
            agents_by_mun[mun_id].append(agent)

        for family in sim.families.values():
            # TODO: sometimes family.region_id is None?
            if family.region_id:
                mun_id = family.region_id[:7]
                families_by_mun[mun_id].append(family)
            else:
                families_by_mun[family.region_id].append(family)

        # Pre-compute firm revenue per municipality in O(n_firms) — avoids O(n_firms × n_muns) loop
        mun_firm_revenue = defaultdict(float)
        for firm in sim.firms.values():
            mun_firm_revenue[firm.region_id[:7]] += firm.revenue

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
            GDP_mun_capita = mun_firm_revenue[mun_id] / mun_pop if mun_pop > 0 else 0
            commuting = sim.stats.update_commuting(mun_families)
            mun_gini = sim.stats.calculate_regional_gini(mun_families)
            mun_house_values = sim.stats.calculate_avg_regional_house_price(mun_families)
            mun_unemployment = sim.stats.update_unemployment(mun_agents)
            region.total_commute = commuting

            families_regional_metrics = sim.stats.calculate_families_metrics(mun_families)

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

            # The two OGU pots (and hence both diag dicts) are keyed on the 6-digit
            # IBGE prefix; mun_id here carries all 7 digits.
            mcmv = sim.funds.mcmv_diag.get(mun_id[:6]) if hasattr(sim.funds, 'mcmv_diag') else None
            if mcmv is None:
                mcmv = sim.funds.blank_mcmv_diag()
            melhorias = sim.funds.melhorias_diag.get(mun_id[:6]) if hasattr(sim.funds, 'melhorias_diag') else None
            if melhorias is None:
                melhorias = sim.funds.blank_melhorias_diag()

            reports.append(
                '%s;%s;%.3f;%d;%.3f;%.4f;%.3f;%.4f;%.5f;%.3f;%.6f;%.6f;%.6f;%.6f;%s;%.6f;%.6f;%.6f'
                ';%d;%d;%d;%.2f;%.2f;%.2f;%d;%d;%d;%d;%d;%d'
                ';%d;%d;%.2f;%.2f;%.2f'
                % (sim.clock.days, mun_id, commuting, mun_pop, mun_gdp, mun_gini, mun_house_values,
                   mun_unemployment, mun_qli, GDP_mun_capita, mun_cumulative_treasure,
                   mun_applied_treasure['equally'],
                   mun_applied_treasure['locally'],
                   mun_applied_treasure['fpm'],
                   licenses,
                   families_regional_metrics['affordability_ratio'],
                   families_regional_metrics['median_permanent_income'],
                   families_regional_metrics['median_affordability'],
                   mcmv['eligible'],
                   mcmv['units_available'],
                   mcmv['units_bought'],
                   mcmv['money_topup'],
                   mcmv['money_start'],
                   mcmv['money_residual'],
                   mcmv['stop_no_eligible'],
                   mcmv['stop_no_units'],
                   mcmv['stop_families_exhausted'],
                   mcmv['stop_budget'],
                   mcmv['stop_indivisible'],
                   mcmv['stop_units_exhausted'],
                   melhorias['eligible'],
                   melhorias['upgrades'],
                   melhorias['money_topup'],
                   melhorias['money_start'],
                   melhorias['money_residual'],
                   ))

        with open(self.regional_path, 'a') as f:
            f.write('\n' + '\n'.join(reports))

    def save_neighbourhood_data(self, sim):
        neighbourhood_families = defaultdict(list)
        neighbourhood_gini = dict()
        neighbourhood_commute = dict()
        for family in sim.families.values():
            neighbourhood_families[family.region_id].append(family)
        for r in neighbourhood_families.keys():
            families = neighbourhood_families[r]
            neighbourhood_gini[r] = sim.stats.calculate_regional_gini(families)
            commute_value = sim.stats.update_commuting(families)
            self.sim.regions[r].total_commute = commute_value
            neighbourhood_commute[r] = commute_value
        with open(self.neighbourhood_path, 'a') as fp:
            for region in sim.regions.values():
                fp.write('%s; %s; %s; %d; %.3f; %.3f; %.3f; %.3f \n' %
                         (sim.clock.days, region.id[:7], region.id, region.pop, region.gdp,
                          region.gdp / region.pop,
                          neighbourhood_commute.get(region.id, 0),
                          neighbourhood_gini.get(region.id, 0)))

    def save_data(self, sim):
        # firms data is necessary for plots,
        # so always save
        # self.save_banks_data(sim)

        for each in conf.RUN['SAVE_DATA']:
            # Skip b/c they are saved anyway above
            if each == 'banks':
                continue
            save_fn = getattr(self, 'save_{}_data'.format(each))
            save_fn(sim)

    def save_firms_data(self, sim):
        day = sim.clock.days
        firms_data = {
            'month': [], 'firm_id': [], 'region_id': [], 'mun_id': [],
            'long': [], 'lat': [], 'total_balance': [], 'number_employees': [],
            'stocks': [], 'amount_produced': [], 'price': [], 'amount_sold': [],
            'revenue': [], 'profit': [], 'wages_paid': [], 'input_cost': [],
            'emissions': [], 'eco_eff': [], 'innov_investment': [], 'sector': [],
            'increase_production': []
        }
        construction_data = {
            'month': [], 'firm_id': [], 'region_id': [], 'mun_id': [],
            'long': [], 'lat': [], 'total_balance': [], 'number_employees': [],
            'stocks': [], 'amount_produced': [], 'price': [], 'amount_sold': [],
            'revenue': [], 'profit': [], 'wages_paid': []
        }
        for firm in sim.firms.values():
            firms_data['month'].append(day)
            firms_data['firm_id'].append(firm.id)
            firms_data['region_id'].append(firm.region_id)
            firms_data['mun_id'].append(firm.region_id[:7])
            firms_data['long'].append(firm.address.x)
            firms_data['lat'].append(firm.address.y)
            firms_data['total_balance'].append(firm.total_balance)
            firms_data['number_employees'].append(firm.num_employees)
            firms_data['stocks'].append(firm.total_quantity)
            firms_data['amount_produced'].append(firm.amount_produced)
            firms_data['price'].append(firm.prices)
            firms_data['amount_sold'].append(firm.amount_sold)
            firms_data['revenue'].append(firm.revenue)
            firms_data['profit'].append(firm.profit)
            firms_data['wages_paid'].append(firm.wages_paid)
            firms_data['input_cost'].append(firm.input_cost)
            firms_data['emissions'].append(firm.last_emissions)
            firms_data['eco_eff'].append(firm.env_efficiency)
            firms_data['innov_investment'].append(firm.inno_inv)
            firms_data['sector'].append(firm.sector)
            firms_data['increase_production'].append(firm.increase_production)
            if firm.sector == 'Construction':
                construction_data['month'].append(day)
                construction_data['firm_id'].append(firm.id)
                construction_data['region_id'].append(firm.region_id)
                construction_data['mun_id'].append(firm.region_id[:7])
                construction_data['long'].append(firm.address.x)
                construction_data['lat'].append(firm.address.y)
                construction_data['total_balance'].append(firm.total_balance)
                construction_data['number_employees'].append(firm.num_employees)
                construction_data['stocks'].append(firm.total_quantity)
                construction_data['amount_produced'].append(len(firm.houses_built))
                construction_data['price'].append(firm.mean_house_price())
                construction_data['amount_sold'].append(firm.n_houses_sold)
                construction_data['revenue'].append(firm.revenue)
                construction_data['profit'].append(firm.profit)
                construction_data['wages_paid'].append(firm.wages_paid)
            firm.reset_amount_sold()

        self._write_parquet('firms', self.firms_path, firms_data)
        if construction_data['month']:
            self._write_parquet('construction', self.construction_path, construction_data)

    def save_agents_data(self, sim):
        day = sim.clock.days
        data = {
            'month': [], 'region_id': [], 'gender': [], 'x': [], 'y': [],
            'id': [], 'age': [], 'qualification': [], 'firm_id': [],
            'family_id': [], 'money': [], 'distance': []
        }
        for agent in sim.agents.values():
            data['month'].append(day)
            data['region_id'].append(agent.region_id)
            data['gender'].append(agent.gender)
            data['x'].append(agent.address.x)
            data['y'].append(agent.address.y)
            data['id'].append(agent.id)
            data['age'].append(agent.age)
            data['qualification'].append(agent.qualification)
            data['firm_id'].append(agent.firm_id)
            data['family_id'].append(agent.family.id)
            data['money'].append(agent.money)
            data['distance'].append(agent.distance)
        self._write_parquet('agents', self.agents_path, data)

    def save_grave_data(self, sim):
        if sim.grave:
            day = sim.clock.days
            data = {
                'month': [], 'region_id': [], 'gender': [], 'x': [], 'y': [],
                'id': [], 'age': [], 'qualification': [], 'firm_id': [],
                'family_id': [], 'money': [], 'utility': [], 'distance': []
            }
            for agent in sim.grave:
                data['month'].append(day)
                data['region_id'].append(agent.region_id)
                data['gender'].append(agent.gender)
                data['x'].append(agent.address.x if agent.address else None)
                data['y'].append(agent.address.y if agent.address else None)
                data['id'].append(agent.id)
                data['age'].append(agent.age)
                data['qualification'].append(agent.qualification)
                data['firm_id'].append(agent.firm_id)
                data['family_id'].append(agent.family.id if agent.family else None)
                data['money'].append(agent.money)
                data['utility'].append(agent.utility)
                data['distance'].append(agent.distance)
            self._write_parquet('grave', self.grave_path, data)
        sim.grave.clear()

    def save_house_data(self, sim):
        day = sim.clock.days
        data = {
            'month': [], 'id': [], 'x': [], 'y': [], 'size': [], 'price': [],
            'rent': [], 'quality': [], 'qli': [], 'on_market': [],
            'family_id': [], 'region_id': [], 'mun_id': []
        }
        for house in sim.houses.values():
            data['month'].append(day)
            data['id'].append(house.id)
            data['x'].append(house.address.x)
            data['y'].append(house.address.y)
            data['size'].append(house.size)
            data['price'].append(house.price)
            data['rent'].append(house.rent_data[0] if house.rent_data else None)
            data['quality'].append(house.quality)
            data['qli'].append(sim.regions[house.region_id].index)
            data['on_market'].append(house.on_market)
            data['family_id'].append(house.family_id)
            data['region_id'].append(house.region_id)
            data['mun_id'].append(house.region_id[:7])
        self._write_parquet('houses', self.houses_path, data)

    def save_family_data(self, sim):
        day = sim.clock.days
        data = {
            'month': [], 'id': [], 'mun_id': [], 'house_price': [],
            'house_rent': [], 'total_wage': [], 'savings': [], 'num_members': []
        }
        for family in sim.families.values():
            data['month'].append(day)
            data['id'].append(family.id)
            data['mun_id'].append(family.region_id[:7])
            data['house_price'].append(family.house.price if family.house else None)
            data['house_rent'].append(family.house.rent_data[0] if family.house and family.house.rent_data else None)
            data['total_wage'].append(family.total_wage())
            data['savings'].append(family.savings)
            data['num_members'].append(family.num_members)
        self._write_parquet('families', self.families_path, data)

    def prepare_dataframe(self, sim):
        """Converts nested dictionary (class range → month → count) into a DataFrame."""
        data = []
        for class_range, months in sim.stats.head_rate.items():
            for month, count in months.items():
                data.append({"month": month, "class_range": class_range, "count": count})
        df = pd.DataFrame(data)

        # Ensure month sorting
        df["month"] = pd.to_datetime(df["month"], errors='coerce').dt.month
        df = df.sort_values("month")
        return df

    def save_head_data(self, sim):
        data = self.prepare_dataframe(sim)
        data.to_csv(self.head_path, index=False)

    def save_banks_data(self, sim):
        bank = sim.central
        active = bank.active_loans()
        n_active = len(active)
        mean_age = sum(l.age for l in active) / n_active if n_active else 0
        p_delinquent = len(bank.delinquent_loans()) / n_active if n_active else 0
        mn, mx, avg = bank.loan_stats_summary()
        data = {
            'month': [sim.clock.days],
            'balance': [bank.balance],
            'active_loans': [n_active],
            'mortgage_rate': [bank.mortgage_rate],
            'p_delinquent_loans': [p_delinquent],
            'mean_loan_age': [mean_age],
            'min_loan': [mn],
            'max_loan': [mx],
            'mean_loan': [avg],
        }
        self._write_parquet('banks', self.banks_path, data)

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