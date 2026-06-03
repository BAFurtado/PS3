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
        self.gov_consumption_parameter = self.sim.regional_market.final_demand['GovernmentConsumption']['Government']
        self.perc_policy_money_spent = 0
        self.allocated_money = 0
        if sim.PARAMS['FPM_DISTRIBUTION']:
            self.fpm = {
                state: pd.read_csv('input/fpm/%s.csv' % state, sep=',', header=0, decimal='.', encoding='latin1')
                for state in self.sim.geo.states_on_process}
        if self.needs_policy_funding():
            self.policy_money = defaultdict(float)
            self.policy_families = defaultdict(list)
            self.temporary_houses = defaultdict(list)

        if sim.PARAMS['POLICY_MCMV'] or sim.PARAMS['POLICY_MELHORIAS']:
            # Collect money from exogenous funding
            self.mcmv = MCMV(sim)

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

        if self.sim.PARAMS['POLICY_MCMV']:
            # MCMV FAIXA 1
            self.policy_money = self.mcmv.update_policy_money(self.sim.clock.year)
            self.allocated_money += sum(self.policy_money.values())
            quantile = self.sim.PARAMS['INCOME_MODALIDADES']['faixa1']
            self.update_policy_families(quantile)
            self.buy_houses_give_to_families()
            # RURAL
            # self.policy_money = self.mcmv.update_policy_money(self.sim.clock.year, 'rural')
            # quantile = self.sim.PARAMS['INCOME_MODALIDADES']['rural']
            # self.update_policy_families(quantile)
            # for mun in self.policy_families.keys():
            #     self.policy_families[mun] = [f for f in self.policy_families[mun] if f.house.rural]
            # self.buy_houses_give_to_families()
        if self.sim.PARAMS['POLICY_MELHORIAS']:
            self.policy_money = self.mcmv.update_policy_money(self.sim.clock.year)
            self.allocated_money += sum(self.policy_money.values())
            quantile = self.sim.PARAMS['INCOME_MODALIDADES']['melhorias']
            self.update_policy_families(quantile)
            for mun in self.policy_families.keys():
                self.policy_families[mun] = [f for f in self.policy_families[mun] if f.house.quality == .5]
            self.apply_house_upgrade()

        if self.sim.PARAMS['POLICY_COEFFICIENT']:
            self.update_policy_families(self.sim.PARAMS['POLICY_COEFFICIENT'])
            if self.sim.PARAMS['POLICIES'] == 'buy':
                self.buy_houses_give_to_families()
            elif self.sim.PARAMS['POLICIES'] == 'rent':
                self.pay_families_rent()
            elif self.sim.PARAMS['POLICIES'] == 'wage':
                self.distribute_funds_to_families()

        if self.allocated_money:
            self.perc_policy_money_spent = self.money_applied_policy / self.allocated_money

        if self.sim.PARAMS['CARBON_TAX_RECYCLING']:
            self.recycle_carbon_tax(self.sim.regions)
        # Resetting lists for next month
        self.allocated_money = 0
        self.policy_families = defaultdict(list)
        self.temporary_houses = defaultdict(list)

    def apply_house_upgrade(self):
        # STRICTLY FROM .5 TO 1
        for mun in self.policy_families.keys():
            self.policy_families[mun] = [f for f in self.policy_families[mun] if
                                         (f.house.family_id == f.id) &
                                         (f.house.quality == .5)]
            for family in self.policy_families[mun]:
                if self.policy_money[mun] > 0:
                    upgrade_cost = family.house.price * self.sim.PARAMS['UPGRADE_COST']
                    if self.policy_money[mun] > upgrade_cost:
                        family.house.quality = 1
                        self.policy_money[mun] -= upgrade_cost
                        self.money_applied_policy += upgrade_cost
                        self.families_subsided += 1
                else:
                    break

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

    def buy_houses_give_to_families(self):
        houses_by_mun = defaultdict(list)
        for firm in self.sim.firms.values():
            if firm.sector == 'Construction':
                for h in firm.houses_for_sale:
                    houses_by_mun[h.region_id[:6]].append(h)
        # Families are sorted in self.policy_families. Buy and give as much as money allows
        for mun in self.policy_money.keys():
            self.temporary_houses[mun] = houses_by_mun.get(mun, [])
            # Sort houses and families by cheapest, poorest.
            # Considering # houses is limited, help as many as possible earlier.
            # Although families in succession gets better and better houses. Then nothing.
            self.temporary_houses[mun] = sorted(self.temporary_houses[mun], key=lambda h: h.price)
            # Exclude families who own any house. Exclusively for renters
            self.policy_families[mun] = [f for f in self.policy_families[mun] if not f.owned_houses]

            for house in self.temporary_houses[mun]:
                # While families to receive houses
                if not self.policy_families[mun]:
                    break
                # While money is good.
                if self.policy_money[mun] <= 0 or house.price >= self.policy_money[mun]:
                    break
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
                self.policy_money[mun] -= house.price
                # Transfer ownership
                self.sim.firms[house.owner_id].houses_for_sale.remove(house)
                # Finish notarial procedures
                house.owner_id = family.id
                house.family_owner = True
                family.owned_houses.append(house)
                house.on_market = 0
                # Move out. Move in
                HousingMarket.make_move(family, house, self.sim)

        # Clean up list for next month
        self.temporary_houses = defaultdict(list)

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

        total_fpm = sum(set(fpm_region.values()))
        for id, region in regions.items():
            mun_code = region.id[:7]
            if total_fpm == 0 or pop_mun_t[mun_code] == 0:
                regional_fpm = 0.0
            else:
                regional_fpm = fpm_region[id] / total_fpm * value * pop_t[id] / pop_mun_t[mun_code]

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

            # BRING treasure from regions to municipalities
            treasure[id] = region.transfer_treasure()

        # QLI: logistic growth driven by municipal economic development.
        # IDHM is a municipal-level statistic, so all regions in the same
        # municipality receive the same update based on municipal GDP per capita.
        for id, region in regions.items():
            m_id = id[:7]
            mun_pop = pop_mun_t[m_id]
            gdp_per_pop = max(0.0, gdp_mun_t[m_id]) / mun_pop if mun_pop > 0 else 0.0
            region.update_qli(gdp_per_pop, self.sim.PARAMS)

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
            