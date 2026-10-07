import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

from markets.housing import HousingMarket
from .geography import STATES_CODES, state_string
from .mcmv_funds import MCMV


class Funds:
    def __init__(self, sim):
        self.sim = sim
        self.families_subsided = 0
        self.money_applied_policy = 0
        self.carbon_tax_recycled_money = 0
        self.mun_gov_firms = defaultdict(list)
        # GOV_REVISED: public revenue per region and channel ('fpm', 'locally', 'equally'), held until
        # settle_government_budget pays the public payroll and purchases and passes the rest on to the regions.
        self.pending_public_money = defaultdict(lambda: defaultdict(float))
        self.gov_consumption_parameter = self.sim.regional_market.final_demand['GovernmentConsumption']['Government']
        # GOV_WAGE_RULE inputs, keyed by 7-digit municipality code as str: observed raw public/private wage ratio
        # ('cempre_ratio') and the share of public jobs by level of government ('premium', GOV_EXTERNAL_FUNDING)
        self.gov_wage_ratio = defaultdict(lambda: 1.0)
        self.gov_levels = defaultdict(lambda: {'federal': 0.0, 'estadual': 0.0, 'municipal': 1.0})
        rule = sim.PARAMS.get('GOV_WAGE_RULE', 'uniform')
        if rule == 'cempre_ratio':
            ratios = pd.read_csv('input/gov_wage_ratio.csv', sep=';')
            self.gov_wage_ratio.update(zip(ratios.cod_mun.astype(str), ratios.ratio))
        if rule == 'premium' or sim.PARAMS.get('GOV_EXTERNAL_FUNDING', False):
            levels = pd.read_csv('input/gov_levels.csv', sep=';')
            self.gov_levels.update({str(r.cod_mun): {'federal': r.federal, 'estadual': r.estadual,
                                                     'municipal': r.municipal} for r in levels.itertuples()})
        # Federal and state pay per municipality as multiples of its ACP's private pay,
        # and the ACP private pay per worker they apply to: this month's until GOV_PAY_BURN_IN + GOV_PAY_BASE_MONTHS
        # settlements, then the mean over the base months, fixed
        self.gov_pay = {}
        self.gov_pay_months = []
        # Each municipality's public investment in real terms over the settlements so far, and its
        # base-months mean once fixed
        self.gov_spending_months = defaultdict(list)
        self.gov_spending_base = {}
        self.gov_pay_reference = None
        pay = pd.read_csv('input/gov_pay.csv', sep=';')
        self.gov_pay = {str(r.cod_mun): {'federal': r.federal, 'estadual': r.estadual} for r in pay.itertuples()}
        # GOV_EXTERNAL_FUNDING: money paid in from outside the ACP for federal and state public staff, cumulative
        self.external_public_funding = 0.0
        self.perc_policy_money_spent = 0
        self.allocated_money = 0
        # Per-municipality diagnostics of the two OGU programmes, refreshed every
        # month and written out by analysis/output.py:save_regional_report. ACP-level
        # perc_policy_money_spent averages the municipalities together and cannot say
        # *why* the programme stopped; these can.
        self.mcmv_diag = {}
        self.melhorias_diag = {}
        # GOV_REVISED: per municipality, last month's public budget, external funding, target payroll, wage paid per
        # worker, payroll and public investment
        self.gov_budget_diag = {}
        if sim.PARAMS['FPM_DISTRIBUTION']:
            self.fpm = {
                state: pd.read_csv('input/fpm/%s.csv' % state, sep=',', header=0, decimal='.', encoding='latin1')
                for state in self.sim.geo.states_on_process}
        if self.needs_policy_funding():
            # Tax-funded pot for the POLICY_COEFFICIENT policies (buy/rent/wage),
            # fed by distribute_fpm/locally/equally. Distinct from the two OGU pots.
            self.policy_money = defaultdict(float)
            self.policy_families = defaultdict(list)
            self.temporary_houses = defaultdict(list)

        if sim.PARAMS['POLICY_MCMV'] or sim.PARAMS['POLICY_MELHORIAS']:
            # Collect money from exogenous funding
            self.mcmv = MCMV(sim)
            # MCMV and melhorias are two distinct programmes, each with its own OGU
            # budget line of the same size. One persistent pot each: topped up every
            # month by `share x GDP / 12` and never reset, so money a municipality
            # could not spend -- typically because the cheapest remaining unit cost
            # more than what was left -- is still there next month.
            self.policy_money_mcmv = defaultdict(float)
            self.policy_money_melhorias = defaultdict(float)

    @staticmethod
    def blank_mcmv_diag():
        """One municipality-month of MCMV allocation diagnostics.

        `money_topup` is this month's OGU allocation, `money_start` the balance
        available to spend (top-up plus whatever carried over) and `money_residual`
        what carries into next month.

        The six stop_* fields are mutually exclusive indicators (exactly one is 1
        whenever the municipality was processed at all), so averaging them across
        runs gives the probability of each stopping reason.
        """
        return {
            'eligible': 0,
            'units_available': 0,
            'units_bought': 0,
            'money_topup': 0.0,
            'money_start': 0.0,
            'money_residual': 0.0,
            'stop_no_eligible': 0,
            'stop_no_units': 0,
            'stop_families_exhausted': 0,
            'stop_budget': 0,
            'stop_indivisible': 0,
            'stop_units_exhausted': 0,
        }

    @staticmethod
    def blank_melhorias_diag():
        """One municipality-month of melhorias allocation diagnostics.

        Same money fields as blank_mcmv_diag. `upgrades` counts houses taken from
        quality .5 to 1 this month.

        The five stop_* fields are mutually exclusive indicators (exactly one is 1
        whenever the municipality was processed at all), so averaging them across
        runs gives the probability of each stopping reason. Money and construction
        capacity are separate constraints: `stop_budget` means the pot could not
        cover a work, `stop_no_capacity` that the money was there but no local
        builder could do the work this month.
        """
        return {
            'eligible': 0,
            'upgrades': 0,
            'money_topup': 0.0,
            'money_start': 0.0,
            'money_residual': 0.0,
            'stop_no_eligible': 0,
            'stop_no_builder': 0,
            'stop_no_capacity': 0,
            'stop_budget': 0,
            'stop_families_exhausted': 0,
        }

    @staticmethod
    def top_up(pot, allocation, diagnostics, blank):
        """Add this month's `allocation` to a persistent programme `pot`.

        The pot keeps its residual from previous months, so the balance a
        municipality starts the month with is carryover plus top-up. Opens one
        diagnostics record per municipality.

        Returns the allocation rather than the balance, because it is the denominator
        of `perc_policy_money_spent`, a budget-execution rate against the month's
        budget line. A balance denominator decays towards zero for any programme whose
        pot accumulates faster than it spends.
        """
        for mun, value in allocation.items():
            pot[mun] += value
        for mun, balance in pot.items():
            diag = blank()
            diag['money_topup'] = allocation.get(mun, 0.0)
            diag['money_start'] = balance
            diagnostics[mun] = diag
        return sum(allocation.values())

    def needs_policy_funding(self):
        return (
                (self.sim.PARAMS['POLICIES'] in ['buy', 'rent', 'wage'] and self.sim.PARAMS['POLICY_COEFFICIENT'] > 0)
                or self.sim.PARAMS.get('POLICY_MCMV')
                or self.sim.PARAMS.get('POLICY_MELHORIAS')
                or self.sim.PARAMS.get('CARBON_TAX_RECYCLING')
        )

    def update_policy_families(self, quantile):
        today = self.sim.clock.days

        families = list(self.sim.families.values())

        # Compute quantile from a temporary contiguous array, then free it immediately
        incomes = np.fromiter((f.permanent_income for f in families), dtype=np.float64, count=len(families))
        quantile_value = np.quantile(incomes, quantile)
        del incomes

        # Group families by region
        families_by_region = defaultdict(list)
        for f in families:
            families_by_region[f.house.region_id].append(f)

        # Register low-income families by region
        for region in self.sim.regions.values():
            eligible = [
                f for f in families_by_region[region.id]
                if f.permanent_income < quantile_value
            ]
            region.registry[today].extend(eligible)

        window_start = today - datetime.timedelta(self.sim.PARAMS['POLICY_DAYS'])

        # Prune old registry entries
        for region in self.sim.regions.values():
            keys_to_delete = [k for k in region.registry if k <= window_start]
            for k in keys_to_delete:
                del region.registry[k]

        # Build policy families
        temp_policy = defaultdict(dict)

        for region in self.sim.regions.values():
            mun = region.id[:6]
            for key, fams in region.registry.items():
                if key > window_start:
                    for f in fams:
                        temp_policy[mun][f.id] = f

        # Convert back to expected structure
        # Convert back to expected structure (SAFE VERSION)
        self.policy_families = defaultdict(list)

        for mun, fams in temp_policy.items():
            self.policy_families[mun] = list(fams.values())

        #  Final filtering
        valid_families = self.sim.families.keys()

        for mun in self.policy_families:
            filtered = [
                f for f in self.policy_families[mun]
                if f.id in valid_families and f.house.region_id[:6] == mun
            ]

            if self.sim.PARAMS['TOTAL_TARGETING_POLICY']:
                filtered.sort(key=lambda f: f.permanent_income)

            self.policy_families[mun] = filtered

    def apply_policies(self):
        if not self.needs_policy_funding():
            # Baseline scenario. Do nothing!
            return
        # Implement policies only after first year of simulation run. Commented for MCMV policy. Existed in 2010.
        # if self.sim.clock.days < self.sim.PARAMS['STARTING_DAY'] + datetime.timedelta(360):
        #     return
        # Reset monthly indicators so stats reflect current month, not cumulative totals
        self.families_subsided = 0
        self.money_applied_policy = 0
        self.mcmv_diag = {}
        self.melhorias_diag = {}
        self.allocated_money = 0

        if self.sim.PARAMS['POLICY_MCMV']:
            # MCMV FAIXA 1
            topped = self.top_up(
                self.policy_money_mcmv, self.mcmv.monthly_allocation(self.sim.clock.year),
                self.mcmv_diag, self.blank_mcmv_diag)
            self.allocated_money += topped
            self.sim.ledger['ogu'] += topped
            quantile = self.sim.PARAMS['INCOME_MODALIDADES']['faixa1']
            self.update_policy_families(quantile)
            self.buy_houses_give_to_families(self.policy_money_mcmv, self.mcmv_diag)
            # RURAL
            # quantile = self.sim.PARAMS['INCOME_MODALIDADES']['rural']
            # self.update_policy_families(quantile)
            # for mun in self.policy_families.keys():
            #     self.policy_families[mun] = [f for f in self.policy_families[mun] if f.house.rural]
            # self.buy_houses_give_to_families(self.policy_money_mcmv, self.mcmv_diag)
        if self.sim.PARAMS['POLICY_MELHORIAS']:
            topped = self.top_up(
                self.policy_money_melhorias, self.mcmv.monthly_allocation(self.sim.clock.year),
                self.melhorias_diag, self.blank_melhorias_diag)
            self.allocated_money += topped
            self.sim.ledger['ogu'] += topped
            quantile = self.sim.PARAMS['MELHORIAS_INCOME_QUANTILE']
            self.update_policy_families(quantile)
            for mun in self.policy_families.keys():
                self.policy_families[mun] = [f for f in self.policy_families[mun] if f.house.quality == .5]
            self.apply_house_upgrade(self.policy_money_melhorias, self.melhorias_diag)

        if self.sim.PARAMS['POLICY_COEFFICIENT']:
            self.allocated_money += sum(self.policy_money.values())
            self.update_policy_families(self.sim.PARAMS['POLICY_COEFFICIENT'])
            if self.sim.PARAMS['POLICIES'] == 'buy':
                self.buy_houses_give_to_families(self.policy_money)
            elif self.sim.PARAMS['POLICIES'] == 'rent':
                self.pay_families_rent()
            elif self.sim.PARAMS['POLICIES'] == 'wage':
                self.distribute_funds_to_families()

        if self.allocated_money:
            self.perc_policy_money_spent = self.money_applied_policy / self.allocated_money

        if self.sim.PARAMS['CARBON_TAX_RECYCLING']:
            self.recycle_carbon_tax(self.sim.regions)
        # Resetting lists for next month
        self.policy_families = defaultdict(list)
        self.temporary_houses = defaultdict(list)

    def apply_house_upgrade(self, policy_money, diagnostics):
        """Melhorias: the municipality contracts a local construction firm to take an
        eligible house from quality .5 to 1. STRICTLY FROM .5 TO 1.

        The works consume construction capacity (`total_quantity`) at the cost
        `ConstructionFirm.plan_house` would charge for the same quality delta on the
        same floor area, so a refurbishment competes with new houses for the same
        production, and the builder is paid for it exactly as it is paid for an MCMV
        acquisition. A house is upgraded only when the money *and* a builder with the
        capacity to do the whole work this month are both available; families that
        miss out stay eligible next month, and the persistent pot carries their money
        with them.
        """
        builders = defaultdict(list)
        for firm in self.sim.firms.values():
            if firm.sector == 'Construction' and not firm.own_account:
                builders[firm.region_id[:6]].append(firm)

        for mun in policy_money.keys():
            diag = diagnostics[mun]
            eligible = [f for f in self.policy_families[mun] if
                        (f.house.family_id == f.id) &
                        (f.house.quality == .5)]
            self.policy_families[mun] = eligible
            diag['eligible'] = len(eligible)

            if not eligible:
                diag['stop_no_eligible'] = 1
            elif not builders[mun]:
                diag['stop_no_builder'] = 1
            else:
                denied_budget, denied_capacity = False, False
                for family in eligible:
                    # The works are priced by the model's own price function: taking
                    # quality .5 to 1 on the same floor area in the same region is
                    # worth `size * .5 * region.index`, i.e. the house's own
                    # pre-upgrade price. UPGRADE_COST is the share of that market
                    # price the state pays.
                    upgrade_cost = family.house.price * self.sim.PARAMS['UPGRADE_COST']
                    if policy_money[mun] <= upgrade_cost:
                        # Houses are of different sizes and the queue is ordered by
                        # income, not by price, so a later family may still be
                        # affordable: keep going rather than break.
                        denied_budget = True
                        continue
                    firm, work_cost = self.hire_builder(builders[mun], family.house)
                    if firm is None:
                        denied_capacity = True
                        continue
                    self.upgrade_house(mun, family.house, firm, work_cost,
                                       upgrade_cost, policy_money)
                    diag['upgrades'] += 1
                if denied_capacity:
                    stop = 'no_capacity'
                elif denied_budget:
                    stop = 'budget'
                else:
                    stop = 'families_exhausted'
                diag['stop_{}'.format(stop)] = 1

            diag['money_residual'] = policy_money[mun]

    def hire_builder(self, builders, house):
        """The municipal construction firm with the most idle production takes the
        job, provided it can do the whole work within the month.

        Returns `(firm, work_cost)`, or `(None, 0)` when no local builder has the
        capacity. Ordering is by capacity alone and ties keep firm creation order, so
        the choice draws nothing from the shared random stream.
        """
        region = self.sim.regions[house.region_id]
        for firm in sorted(builders, key=lambda f: -f.total_quantity):
            # Same cost formula as ConstructionFirm.plan_house, for a quality delta of
            # .5 over the floor area the house already has. Firms that have never
            # planned a house have not drawn a productivity yet and are costed at 1.
            work_cost = self.sim.house_values.upgrade_cost(
                region.id, house.size, firm.productivity or self.sim.house_values.mean_productivity) / firm.prices

            if firm.total_quantity >= work_cost:
                return firm, work_cost
        return None, 0

    def upgrade_house(self, mun, house, firm, work_cost, upgrade_cost, policy_money):
        """Municipality pays `firm` for the works and the house moves to quality 1."""
        # The works consume the production a new house would have consumed
        firm.total_quantity -= work_cost
        # Mirrors the MCMV acquisition leg: the region taxes the transaction and the
        # builder books the rest as revenue accrued over the construction cash-flow
        # window, which is what reaches wages through ConstructionFirm.wage_base.
        taxes = upgrade_cost * self.sim.PARAMS['TAX_ESTATE_TRANSACTION']
        self.sim.regions[house.region_id].collect_taxes(taxes, 'transaction')
        firm.update_balance(upgrade_cost - taxes,
                            self.sim.PARAMS['CONSTRUCTION_ACC_CASH_FLOW'],
                            self.sim.clock.days)
        house.quality = 1
        policy_money[mun] -= upgrade_cost
        self.money_applied_policy += upgrade_cost
        self.families_subsided += 1

    def pay_families_rent(self):
        for mun in self.policy_money.keys():
            self.policy_families[mun] = [f for f in self.policy_families[mun] if not f.owned_houses]
            for family in self.policy_families[mun]:
                if family.house.rent_data:
                    if self.policy_money[mun] > 0:
                        if family.house.rent_data[0] * 24 < self.policy_money[mun]:
                            if not family.rent_voucher:
                                # Paying rent for a given number of months, independent of rent value.
                                family.rent_voucher = 24
                                self.policy_money[mun] -= family.house.rent_data[0] * 24
                                self.money_applied_policy += family.house.rent_data[0] * 24
                                self.families_subsided += 1
                    else:
                        break

    def distribute_funds_to_families(self):
        for mun in self.policy_money.keys():
            if self.policy_families[mun] and self.policy_money[mun] > 0:
                # Registering subsidies
                self.money_applied_policy += self.policy_money[mun]
                self.families_subsided += len(self.policy_families[mun])
                # Amount is proportional to available funding and families
                amount = self.policy_money[mun] / len(self.policy_families[mun])
                [f.update_balance(amount) for f in self.policy_families[mun]]
                # Reset fund because it has been totally expended.
                self.policy_money[mun] = 0

    def buy_houses_give_to_families(self, policy_money, diagnostics=None):
        houses_by_mun = defaultdict(list)
        for firm in self.sim.firms.values():
            if firm.sector == 'Construction' and not firm.pool:
                for h in firm.houses_for_sale:
                    houses_by_mun[h.region_id[:6]].append(h)
        # Families are sorted in self.policy_families. Buy and give as much as money allows
        for mun in policy_money.keys():
            # The POLICY_COEFFICIENT 'buy' policy shares this loop but is a different
            # programme, so it reports no MCMV diagnostics.
            diag = diagnostics[mun] if diagnostics is not None else self.blank_mcmv_diag()

            self.temporary_houses[mun] = houses_by_mun.get(mun, [])
            # Sort houses and families by cheapest, poorest.
            # Considering # houses is limited, help as many as possible earlier.
            # Although families in succession gets better and better houses. Then nothing.
            self.temporary_houses[mun] = sorted(self.temporary_houses[mun], key=lambda h: h.price)
            # Exclude families who own any house. Exclusively for renters
            self.policy_families[mun] = [f for f in self.policy_families[mun] if not f.owned_houses]

            diag['eligible'] = len(self.policy_families[mun])
            diag['units_available'] = len(self.temporary_houses[mun])

            if not self.policy_families[mun]:
                diag['stop_no_eligible'] = 1
            elif not self.temporary_houses[mun]:
                diag['stop_no_units'] = 1
            else:
                # Optimistic default: only reached if the unit list runs out
                stop = 'units_exhausted'
                for house in self.temporary_houses[mun]:
                    # While families to receive houses
                    if not self.policy_families[mun]:
                        stop = 'families_exhausted'
                        break
                    # While money is good. Budget exhaustion and an indivisible
                    # residual (money left, but the cheapest remaining unit costs
                    # more than what remains) are separate failures -- the second
                    # is a lumpiness problem, not a scarcity one.
                    if policy_money[mun] <= 0:
                        stop = 'budget'
                        break
                    if house.price >= policy_money[mun]:
                        stop = 'indivisible'
                        break
                    self._buy_house_for_family(mun, house, policy_money)
                    diag['units_bought'] += 1
                diag['stop_{}'.format(stop)] = 1

            diag['money_residual'] = policy_money[mun]

        # Clean up list for next month
        self.temporary_houses = defaultdict(list)

    def _buy_house_for_family(self, mun, house, policy_money):
        """Municipality buys `house` from its construction firm and hands it to the
        poorest remaining eligible family in `mun`."""
        # Getting poorest family first, given permanent income
        family = self.policy_families[mun].pop(0)
        # Transaction taxes help reduce the price of the bulk buying by the municipality
        taxes = house.price * self.sim.PARAMS['TAX_ESTATE_TRANSACTION']
        self.sim.regions[house.region_id].collect_taxes(taxes, 'transaction')
        # Register subsidies
        self.money_applied_policy += house.price
        self.families_subsided += 1
        # Pay construction company
        self.sim.firms[house.owner_id].update_balance(house.price - taxes,
                                                      self.sim.PARAMS['CONSTRUCTION_ACC_CASH_FLOW'],
                                                      self.sim.clock.days)
        # Deduce from municipality fund
        policy_money[mun] -= house.price
        # Transfer ownership
        self.sim.firms[house.owner_id].houses_for_sale.remove(house)
        # Finish notarial procedures
        house.owner_id = family.id
        house.family_owner = True
        family.owned_houses.append(house)
        house.on_market = 0
        # Move out. Move in
        HousingMarket.make_move(family, house, self.sim)

    def distribute_fpm(self, value, regions, pop_t, pop_mun_t, year):
        """Calculate proportion of FPM per region, in relation to the total of all regions.
        Value is the total value of FPM to distribute"""
        if float(year) >= 2024:
            year = str(2024)

        # Dictionary that keeps actual FPM received to be used as a proportion parameter
        # to simulated FPM to be distributed
        fpm_region = {}
        states_numbers = [state_string(state, STATES_CODES) for state in self.sim.geo.states_on_process]
        for i, state in enumerate(self.sim.geo.states_on_process):
            for id, region in regions.items():
                if region.id[:2] == states_numbers[i]:
                    mun_code = region.id[:7]
                    fpm_region[id] = self.fpm[state][(self.fpm[state].ano == float(year)) &
                                                     (self.fpm[state].cod == float(mun_code))].fpm.iloc[0]

        # One value per municipality, and only municipalities with people, which are the ones paid below. It used to be
        # sum(set(values)), which also merged municipalities in the same FPM band (equal values), so the shares summed
        # to more than one and FPM money was created (#41)
        fpm_mun = {region_id[:7]: v for region_id, v in fpm_region.items()}
        total_fpm = sum(v for mun, v in fpm_mun.items() if pop_mun_t[mun] > 0)
        for id, region in regions.items():
            mun_code = region.id[:7]
            if total_fpm == 0 or pop_mun_t[mun_code] == 0:
                regional_fpm = 0.0
            else:
                regional_fpm = fpm_region[id] / total_fpm * value * pop_t[id] / pop_mun_t[mun_code]

            if self.sim.PARAMS.get('GOV_REVISED', False):
                self.pending_public_money[id]['fpm'] += regional_fpm
                continue

            # Dividing government investment between intermediate consumption and own consumption
            gov_firms_money = (1 - self.gov_consumption_parameter) * regional_fpm
            [f.government_transfer(gov_firms_money * f.budget_proportion) for f in self.mun_gov_firms[mun_code]]
            regional_fpm = self.gov_consumption_parameter * regional_fpm

            # Separating money for policy
            if self.needs_policy_funding():
                self.policy_money[mun_code] += regional_fpm * self.sim.PARAMS['POLICY_COEFFICIENT']
                regional_fpm *= 1 - self.sim.PARAMS['POLICY_COEFFICIENT']

            region.update_applied_taxes(regional_fpm, 'fpm')

    def locally(self, value, regions, mun_code, pop_t, pop_mun_t):
        for mun in mun_code.keys():
            for id_ in mun_code[mun]:
                amount = value[mun] * pop_t[id_] / pop_mun_t[mun] if pop_mun_t[mun] > 0 else 0.0
                if self.sim.PARAMS.get('GOV_REVISED', False):
                    self.pending_public_money[id_]['locally'] += amount
                    continue
                # Dividing government investment between intermediate consumption and own consumption
                # Check whether there are gov. firms in this municipality at all.
                # When there are no firms, amount is unchanged and goes all to policies and infrastructure
                if self.mun_gov_firms[mun]:
                    firms_here = [f for f in self.mun_gov_firms[mun] if f.region_id == id_]
                    if firms_here:
                        gov_firms_money = (1 - self.gov_consumption_parameter) * amount
                        [f.government_transfer(gov_firms_money * f.budget_proportion)
                         for f in list(self.mun_gov_firms[mun])]
                        amount = self.gov_consumption_parameter * amount

                # Separating money for policy
                if self.needs_policy_funding():
                    self.policy_money[mun] += amount * self.sim.PARAMS['POLICY_COEFFICIENT']
                    amount *= 1 - self.sim.PARAMS['POLICY_COEFFICIENT']

                regions[id_].update_applied_taxes(amount, 'locally')

    def equally(self, value, regions, pop_t, pop_total):
        if self.sim.PARAMS.get('PUBLIC_TAXES_OUT', False):
            # Federal and state revenue raised in the ACP leaves it; federal and state staff are paid from outside
            # (GOV_EXTERNAL_FUNDING)
            self.sim.ledger['public_taxes_out'] -= value
            return
        if self.sim.PARAMS.get('GOV_REVISED', False):
            for id in regions:
                self.pending_public_money[id]['equally'] += value * pop_t[id] / pop_total if pop_total > 0 else 0.0
            return
        # Dividing government investment between intermediate consumption and own consumption
        gov_firms_money = (1 - self.gov_consumption_parameter) * value
        value = self.gov_consumption_parameter * value
        for mun_code in self.mun_gov_firms:
            [f.government_transfer(gov_firms_money * f.budget_proportion) for f in self.mun_gov_firms[mun_code]]

        for id, region in regions.items():
            amount = value * pop_t[id] / pop_total if pop_total > 0 else 0.0
            # Separating money for policy
            if self.needs_policy_funding():
                self.policy_money[id[:7]] += amount * self.sim.PARAMS['POLICY_COEFFICIENT']
                amount *= 1 - self.sim.PARAMS['POLICY_COEFFICIENT']

            region.update_applied_taxes(amount, 'equally')

    def invest_taxes(self, year, bank_taxes):
        # The part of final demand that is not consumed by the government itself is applied in the intermediate
        # market as government purchase. Thus, part of the budget of government following final demand table is
        # distributed at GovernmentFirms to acquire products in the market

        # Setting number within firm that represent the part of the budget and
        # Updating dictionary of government firms
        gov_firms = [f for f in self.sim.firms.values() if f.sector == 'Government']
        for mun_code in self.sim.geo.mun_codes:
            gov_firms_here = [f for f in gov_firms if f.region_id[:7] == str(mun_code)]
            firms_num_employees = [f.num_employees for f in gov_firms_here]
            total_employment = sum(firms_num_employees)
            if total_employment == 0:
                for f in gov_firms_here:
                    f.assign_proportion(0)
            else:
                for f, i in zip(gov_firms_here, firms_num_employees):
                    f.assign_proportion(i / total_employment)
            self.mun_gov_firms[mun_code] = gov_firms_here

        # Collect and UPDATE pop_t-1 and pop_t
        regions = self.sim.regions
        pop_t_minus_1, pop_t = {}, {}
        pop_mun_minus = defaultdict(int)
        pop_mun_t = defaultdict(int)
        gdp_mun_t = defaultdict(float)
        spend_mun_t = defaultdict(float)
        treasure = defaultdict(dict)

        for id, region in regions.items():
            prev_pop = region.pop
            pop_t_minus_1[id] = prev_pop
            pop_mun_minus[id[:7]] += prev_pop
            # Update
            new_pop = self.sim.reg_pops.get(id, 0)
            region.pop = new_pop
            pop_t[id] = new_pop
            pop_mun_t[id[:7]] += new_pop
            gdp_mun_t[id[:7]] += region.gdp
            # Public money applied here since the last call to this method, i.e. the
            # previous month's distribution. Cleared as it is read, so this month's
            # distribution (below) accumulates into a fresh flow.
            spend_mun_t[id[:7]] += region.take_applied_flow()

            # BRING treasure from regions to municipalities
            treasure[id] = region.transfer_treasure()

        # QLI: logistic growth driven by municipal economic development and, at
        # QLI_TAX_WEIGHT > 0, by public spending per capita. IDHM is a municipal-level
        # statistic and one municipality is one administration with one budget, so all
        # regions in the same municipality receive the same update.
        for id, region in regions.items():
            m_id = id[:7]
            mun_pop = pop_mun_t[m_id]
            if mun_pop > 0:
                gdp_per_pop = max(0.0, gdp_mun_t[m_id]) / mun_pop
                spend_per_pop = max(0.0, spend_mun_t[m_id]) / mun_pop
            else:
                gdp_per_pop, spend_per_pop = 0.0, 0.0
            region.update_qli(gdp_per_pop, spend_per_pop, self.sim.PARAMS)

        v_local = defaultdict(float)
        # Every month taxes to distribute start from 0
        v_equal = 0.0
        # All taxes charged from other regions return back to the metropolis
        v_equal += self.sim.external.collect_transfer_consumption_tax()

        if self.sim.PARAMS['ALTERNATIVE0']:
            # Dividing proportion of consumption into equal and local (state, municipality)
            # And adding local part of consumption plus transaction and property to local
            v_equal += sum([treasure[key]['consumption'] for key in treasure.keys()]) * \
                      self.sim.PARAMS['TAXES_STRUCTURE']['consumption_equal']
            mun_code = self.sim.mun_to_regions
            for mun in mun_code.keys():
                v_local[mun] += sum(treasure[r]['consumption'] for r in mun_code[mun]) * \
                                (1 - self.sim.PARAMS['TAXES_STRUCTURE']['consumption_equal'])
                v_local[mun] += sum(treasure[r]['transaction'] for r in mun_code[mun])
                v_local[mun] += sum(treasure[r]['property'] for r in mun_code[mun])
            # The only case in which local funds are distributed
            self.locally(v_local, regions, mun_code, pop_t, pop_mun_t)
        else:
            for each in ['consumption', 'property', 'transaction']:
                v_equal += sum([treasure[key][each] for key in treasure.keys()])

        if self.sim.PARAMS['FPM_DISTRIBUTION']:
            v_fpm = (sum([treasure[key]['labor'] for key in treasure.keys()]) +
                     sum([treasure[key]['firm'] for key in treasure.keys()]))
            self.distribute_fpm(v_fpm * self.sim.PARAMS['TAXES_STRUCTURE']['fpm'], regions, pop_t, pop_mun_t, year)
            v_equal += v_fpm * (1 - self.sim.PARAMS['TAXES_STRUCTURE']['fpm'])
        else:
            v_equal += (sum([treasure[key]['labor'] for key in treasure.keys()]) +
                        sum([treasure[key]['firm'] for key in treasure.keys()]))
        # Taxes charged from interests paid by the bank are equally distributed
        v_equal += bank_taxes
        self.equally(v_equal, regions, pop_t, sum(pop_mun_t.values()))
        if self.sim.PARAMS.get('GOV_REVISED', False):
            self.settle_government_budget(regions)

    def real_public_spending(self, mun, investment):
        """After GOV_PAY_BURN_IN + GOV_PAY_BASE_MONTHS settlements, a municipality's public
        investment is its base months' mean in real terms (deflated by the average goods price) at this month's
        price; the difference from what its budget left comes from (or goes to) outside the ACP"""
        params = self.sim.PARAMS
        price = self.sim.avg_prices if self.sim.avg_prices > 0 else 1.0
        months = self.gov_spending_months[mun]
        if mun not in self.gov_spending_base:
            months.append(investment / price)
            if len(months) >= params['GOV_PAY_BURN_IN'] + params['GOV_PAY_BASE_MONTHS']:
                self.gov_spending_base[mun] = float(np.mean(months[params['GOV_PAY_BURN_IN']:]))
            return investment
        target = self.gov_spending_base[mun] * price
        self.external_public_funding += target - investment
        self.sim.ledger['public_transfers'] += target - investment
        return target

    def national_pay_reference(self, acp_wage):
        """The ACP private pay per worker federal and state pay multiply. This month's
        during GOV_PAY_BURN_IN + GOV_PAY_BASE_MONTHS settlements, then the base months' mean."""
        if self.gov_pay_reference is not None:
            return self.gov_pay_reference
        params = self.sim.PARAMS
        self.gov_pay_months.append(acp_wage)
        if len(self.gov_pay_months) >= params['GOV_PAY_BURN_IN'] + params['GOV_PAY_BASE_MONTHS']:
            self.gov_pay_reference = float(np.mean(self.gov_pay_months[params['GOV_PAY_BURN_IN']:]))
            return self.gov_pay_reference
        return acp_wage

    def settle_government_budget(self, regions):
        """GOV_REVISED: balanced budget. A municipality's public revenue (its FPM, local taxes and share of the taxes
        divided equally) is spent, in order, on: (1) its public payroll, set by GOV_WAGE_RULE ('premium': the
        municipal staff at private pay per unit of qualification, for the qualification Government employs;
        'cempre_ratio' or 'uniform': a ratio times the mean private wage per worker); (2) government purchases, in the input-output ratio of goods to own output in government
        consumption, and the inputs of public production, in the Government column's input share of output;
        (3) policy money (POLICY_COEFFICIENT); (4) public investment with the rest, bought in the input-output FBCF
        shares. (2) and (4) go to funds the Government firms spend on goods (GovernmentFirm.spend_fund,
        buy_inputs), never their start-up capital. If (1) and (2) cost more than the budget, GOV_EXTERNAL_FUNDING
        pays the shortfall from outside the ACP up to the non-municipal share of that cost (federal and state staff
        are paid from national and state revenue); whatever is still short scales the payroll down.
        Federal and state staff are paid the observed multiple of private pay for each level (national_pay_reference),
        fixed in real terms after the base months, municipal staff local pay times one plus GOV_PREMIUM_MUNICIPAL
        ('premium') or the ratio; the outside funding is capped at the federal and state staff's cost. Public
        investment is held in real terms after the base months (real_public_spending).
        (3) and (4) are also recorded as the regions' applied public money, which the QLI fiscal leg reads.
        Nothing is created or lost except the external inflow, counted in external_public_funding. A municipality
        without Government firms has its purchases and investment spent by the ACP's Government firms. The old path
        gave each municipality's firms the whole 'equally' share (x number of municipalities), none of the FPM and
        local shares (str/int key mismatch), and destroyed the regions' share."""
        params = self.sim.PARAMS
        gcp = self.gov_consumption_parameter
        goods_per_wage = (1 - gcp) / gcp
        # Public production inputs per unit of payroll: the Government column of the technical matrices (local and
        # external) is the input share of public output, which is valued at cost (payroll + inputs)
        market = self.sim.regional_market
        input_share = float(market._tech_np['Government'].sum() + market._ext_local_np['Government'].sum())
        inputs_per_wage = input_share / (1 - input_share)

        # Reference private wage: last month's wage bill per worker, and per unit of wage weight (the split
        # make_payment uses), of staffed, paying non-Government firms
        alpha = params['PRODUCTIVITY_EXPONENT']
        rule = params.get('GOV_WAGE_RULE', 'uniform')
        bill, heads, quals = defaultdict(float), defaultdict(int), defaultdict(float)
        for f in self.sim.firms.values():
            if f.sector != 'Government' and not f.own_account and f.num_employees > 0 and f.wages_paid > 0:
                bill[f.region_id[:7]] += f.wages_paid
                heads[f.region_id[:7]] += f.num_employees
                quals[f.region_id[:7]] += f.total_wage_weight(alpha)
        acp_wage = sum(bill.values()) / sum(heads.values()) if heads else 0.0
        acp_unit = sum(bill.values()) / sum(quals.values()) if quals else 0.0
        all_gov = [f for firms in self.mun_gov_firms.values() for f in firms]
        premium_mun = params.get('GOV_PREMIUM_MUNICIPAL', 0.0)
        per_wage = 1 + goods_per_wage + inputs_per_wage
        reference = self.national_pay_reference(acp_wage)

        by_mun = defaultdict(list)
        for id in self.pending_public_money:
            by_mun[id[:7]].append(id)
        for mun, ids in by_mun.items():
            budget = max(0.0, sum(sum(self.pending_public_money[id].values()) for id in ids))
            firms = self.mun_gov_firms[int(mun)]
            staff = sum(f.num_employees for f in firms)
            private_wage = bill[mun] / heads[mun] if heads[mun] else acp_wage
            levels = self.gov_levels[mun]
            # Pay per worker of the federal and state staff, weighted by their share of public jobs
            pay = self.gov_pay.get(mun, {'federal': 0.0, 'estadual': 0.0})
            pay_out = reference * sum(levels[k] * pay[k] for k in ('federal', 'estadual'))
            outside = pay_out * staff
            if rule == 'premium':
                unit = bill[mun] / quals[mun] if quals[mun] else acp_unit
                qual = sum(f.total_wage_weight(alpha) for f in firms)
                w_mun = levels['municipal'] * (1 + premium_mun)
                offer = w_mun * private_wage + pay_out
                target = w_mun * unit * qual + outside
            else:
                ratio = params['GOV_WAGE_RATIO'] * (self.gov_wage_ratio[mun] if rule == 'cempre_ratio' else 1.0)
                offer = ratio * levels['municipal'] * private_wage + pay_out
                target = offer * staff
            # Federal and state staff are paid from outside the ACP when the municipality's budget falls short
            need = target * per_wage
            external = 0.0
            if params.get('GOV_EXTERNAL_FUNDING', False) and need > budget:
                cap = outside * per_wage
                external = min(need - budget, cap)
                self.external_public_funding += external
                self.sim.ledger['public_transfers'] += external
            available = budget + external
            payroll = target * min(1.0, available / need) if need > 0 else 0.0
            wage = payroll / staff if staff > 0 else offer
            purchases = payroll * goods_per_wage
            inputs = payroll * inputs_per_wage
            for f in firms:
                f.budget_first = True
                # Per-worker pay (wage_base, make_payment) and the offer job seekers see (offer_wage); empty firms
                # offer the same, so they can be staffed again
                f.public_wage = wage
                f.public_offer = offer * min(1.0, available / need) if need > 0 else offer
                if staff > 0 and f.num_employees > 0:
                    f.government_transfer(payroll * f.num_employees / staff)
                    f.purchase_fund += purchases * f.num_employees / staff
                    f.input_fund += inputs * f.num_employees / staff
            rest = max(0.0, available - payroll - purchases - inputs)
            share = min(1.0, rest / budget) if budget > 0 else 0.0
            investment = 0.0
            for id in ids:
                for key, amount in self.pending_public_money[id].items():
                    amount *= share
                    if self.needs_policy_funding():
                        self.policy_money[mun] += amount * params['POLICY_COEFFICIENT']
                        amount *= 1 - params['POLICY_COEFFICIENT']
                    regions[id].update_applied_taxes(amount, key)
                    investment += amount
            investment = self.real_public_spending(mun, investment)
            self.gov_budget_diag[mun] = dict(budget=budget, external=external, target=target, wage=wage, staff=staff,
                                             payroll=payroll, investment=investment, outside=outside)

            spenders = firms or all_gov
            if spenders:
                for f in spenders:
                    f.investment_fund += investment / len(spenders)
        self.pending_public_money = defaultdict(lambda: defaultdict(float))

    def recycle_carbon_tax(self,regions):
        # group families by municipality using existing regional structure
        families_by_mun = defaultdict(list)
        for f in self.sim.families.values():
            families_by_mun[f.region_id[:7]].append(f)

        for mun, region_ids in self.sim.mun_to_regions.items():
            total_emissions = sum(regions[rid].treasure["emissions"] for rid in region_ids)
            families = families_by_mun.get(mun, [])
            if total_emissions <= 0 or not families:
                continue

            incomes = [f.permanent_income for f in families]
            threshold = np.percentile(incomes, self.sim.PARAMS['CARBON_RECYCLING_QUANTILE'] * 100)
            
            recipients = [f for f in families if f.permanent_income <= threshold]
            if not recipients:
                continue

            carbon_money = 0
            for region_id in region_ids:
                region_money = 0.8 * regions[region_id].treasure["emissions"]
                carbon_money += region_money
                regions[region_id].collect_taxes(-region_money, "emissions")

            self.carbon_tax_recycled_money += carbon_money
            amount = carbon_money / len(recipients)
            for f in recipients:
                f.update_balance(amount)
            