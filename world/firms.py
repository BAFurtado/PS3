from collections import defaultdict

import numpy as np
import pandas as pd


class FirmData:
    """ Firm growth is estimated from a monthly value of growth observed between the years of 2000 and 2012 """
    def __init__(self, year):
        # Using APs code of year 2000 (they are not compatible with year 2010 APs)
        # If year == 2000, data refers to years 2002 and 2012
        # If year == 2010, data refers to years 2010 and 2017
        self.num_emp_t0 = self._load(f'input/firms_by_APs{year}_t0_full.csv')
        self.num_emp_t1 = self._load(f'input/firms_by_APs{year}_t1_full.csv')

        self.deltas = {}
        self.avg_monthly_deltas = {}
        for mun_code, num_emp_t0 in self.num_emp_t0.items():
            num_emp_t1 = self.num_emp_t1[mun_code]
            delta = num_emp_t1 - num_emp_t0
            self.deltas[mun_code] = delta
            if year == 2000:
                num_months = 12 * 10
            else:
                num_months = 12 * 11
            self.avg_monthly_deltas[mun_code] = delta/num_months

    def _load(self, fname):
        """ Returns the sum of firms of each AP by municipality (all APs summed) """
        num_emp_aps = pd.read_csv(fname, sep=';')
        num_emp = defaultdict(int)
        for idx, row in num_emp_aps.iterrows():
            mun_code = int(str(row['AP'])[:7])
            num_emp[mun_code] += row['num_firms']
            num_emp[int(row['AP'])] = row['num_firms']
        return num_emp


def firm_growth(sim):
    """ Create new firms according to average historical growth
        Location within the municipality is more likely on regions with growth of profit and employees
        """

    # Group firms by region
    firms_by_region = defaultdict(list)
    for firm in sim.firms.values():
        firms_by_region[firm.region_id].append(firm)

    # For each municipality
    for mun_code, regions in sim.mun_to_regions.items():
        # Get growth based on historical data
        growth = sim.generator.firm_data.avg_monthly_deltas[int(mun_code)] * sim.PARAMS['PERCENTAGE_ACTUAL_POP']
        # Round the value and use the remainder as probability
        growth = round(growth)+int(sim.seed_np.random()<growth-round(growth))

        # Ignoring shrinkage for now
        if growth <= 0:
            continue

        # Calculate average profit and number of employees for firms in each region
        avg_profit, avg_n_emp = {}, {}
        for region_id in regions:
            firms = firms_by_region[region_id]
            if firms:
                # keep non-negative for probabilities
                avg_profit[region_id] = max(0, sum(f.profit for f in firms)/len(firms))
                avg_n_emp[region_id] = max(0, sum(f.num_employees for f in firms)/len(firms))
            else:
                avg_profit[region_id] = 0
                avg_n_emp[region_id] = 0

        # Compute probabilities that a firm starts in a region, based on that regions' average
        # profit and number of employees
        region_ps = []
        sum_profit = sum(avg_profit.values())
        sum_n_emp = sum(avg_n_emp.values())
        for region_id in regions:
            if sum_profit == 0 and sum_n_emp == 0:
                # Small non-zero probability
                region_ps.append(0.0001)
            else:
                # Equally weight probability from profit and number of employees
                p_profit = avg_profit[region_id]/sum_profit if sum_profit != 0 else 0
                p_n_emp = avg_n_emp[region_id]/sum_n_emp if sum_n_emp != 0 else 0
                region_ps.append((p_profit + p_n_emp)/2)

        # Normalize probabilities
        region_ps = region_ps / np.sum(region_ps)

        # For each new firm, randomly select its region based on the probabilities we computed
        # and then create the new firm
        for _ in range(growth):
            region_id = sim.seed_np.choice(regions, size=1, replace=True, p=region_ps)[0]
            region = sim.regions[region_id]
            if sim.PARAMS.get('FIRM_CAPITAL_MONTHS', 0) > 0:
                fund_entrant(sim, region)
                continue
            firm = list(sim.generator.create_firms(1, region).values())[0]
            sim.firms[firm.id] = firm
            sim.ledger['firm_entry'] += firm.total_balance


def project_floor(sim):
    """Cost of one median project (ConstructionFirm.plan_house) at the dearest license price: its land (LOT_COST
    share) plus the building cost it advances as wages before the first sale. Fixed at start-up, like the other
    capital scales."""
    if not hasattr(sim, '_project_floor'):
        costs = [h.size * h.quality for h in sim.houses.values()]
        licence = max(r.license_price for r in sim.regions.values())
        cost = licence * float(np.median(costs)) if costs else 0.0
        sim._project_floor = cost * (sim.PARAMS['LOT_COST'] + 1 / sim.PARAMS['HOUSE_PRODUCTION_ADEQUACY'])
    return sim._project_floor


def capital_need(sim, sector, capacity):
    """FIRM_CAPITAL_MONTHS of monthly cost; a builder CONSTRUCTION_CAPITAL_MONTHS, and at least one median project"""
    need = sim.PARAMS['FIRM_CAPITAL_MONTHS'] * capacity
    if sector == 'Construction':
        need = max(sim.PARAMS['CONSTRUCTION_CAPITAL_MONTHS'] * capacity, project_floor(sim))
    return need


def sector_capacity(sim):
    """Median capacity value of the staffed firms of each sector, the scale of a new firm's monthly cost"""
    pe, pd_ = sim.PARAMS['PRODUCTIVITY_EXPONENT'], sim.PARAMS['PRODUCTIVITY_MAGNITUDE_DIVISOR']
    values = defaultdict(list)
    for f in sim.firms.values():
        v = f.capacity_value(pe, pd_)
        if v > 0:
            values[f.sector].append(v)
    return {k: float(np.median(v)) for k, v in values.items()}


def size_initial_capital(sim):
    """After start-up hiring: FIRM_CAPITAL_MONTHS > 0 sizes each firm's capital to its months of cost (unstaffed
    firms: the sector median); GOV_REVISED leaves Government with none, as it spends only its budget."""
    months = sim.PARAMS.get('FIRM_CAPITAL_MONTHS', 0)
    gov_revised = sim.PARAMS.get('GOV_REVISED', False)
    if months <= 0 and not gov_revised:
        return
    pe, pd_ = sim.PARAMS['PRODUCTIVITY_EXPONENT'], sim.PARAMS['PRODUCTIVITY_MAGNITUDE_DIVISOR']
    medians = sector_capacity(sim) if months > 0 else {}
    for f in sim.firms.values():
        if gov_revised and f.sector == 'Government':
            f.total_balance = 0.0
            continue
        if months > 0:
            capacity = f.capacity_value(pe, pd_) or medians.get(f.sector, 0.0)
            f.total_balance = capital_need(sim, f.sector, capacity)
            f.cold_start_share = 1 / months


def surplus(sim, firm, pe, pd_):
    """Capital above the firm's own buffer"""
    return max(0.0, firm.total_balance - capital_need(sim, firm.sector, firm.capacity_value(pe, pd_)))


def pay_profit_shares(sim):
    """FIRM_PAYOUT 'staff': each private firm pays FIRM_PAYOUT_RATE of its cash above its capital buffer (a builder's
    cash net of wages already owed) to its staff"""
    for agent in sim.agents.values():
        agent.last_profit_share = 0.0
    pe, pd_ = sim.PARAMS['PRODUCTIVITY_EXPONENT'], sim.PARAMS['PRODUCTIVITY_MAGNITUDE_DIVISOR']
    rate = sim.PARAMS['FIRM_PAYOUT_RATE']
    paid = 0.0
    for firm in sim.firms.values():
        if firm.sector == 'Government':
            continue
        cash = firm.free_cash() if firm.sector == 'Construction' else firm.total_balance
        paid += firm.pay_profit_share(cash - capital_need(sim, firm.sector, firm.capacity_value(pe, pd_)), rate, pe)
    sim.profit_share_paid = paid


def fund_entrant(sim, region):
    """FIRM_CAPITAL_MONTHS > 0: a new firm enters in a sector drawn from the RAIS shares only if that sector's
    incumbents hold, above their own buffers, the capital it needs; they pay in proportion to their surplus."""
    p = sim.generator.sector_shares()
    if sim.PARAMS.get('GOV_REVISED', False):
        p = p.drop('Government', errors='ignore')
        p = p / p.sum()
    sector = sim.seed_np.choice(list(p.index), p=list(p.values))
    pe, pd_ = sim.PARAMS['PRODUCTIVITY_EXPONENT'], sim.PARAMS['PRODUCTIVITY_MAGNITUDE_DIVISOR']
    capacity = sector_capacity(sim).get(sector, 0.0)
    incumbents = [f for f in sim.firms.values() if f.sector == sector]
    surpluses = [surplus(sim, f, pe, pd_) for f in incumbents]
    need = capital_need(sim, sector, capacity)
    available = sum(surpluses)
    if need <= 0 or available < need:
        sim.firm_entry_unfunded += 1
        return None
    firm = list(sim.generator.create_firms(1, region, firm_sectors=[sector]).values())[0]
    for f, s in zip(incumbents, surpluses):
        f.total_balance -= need * s / available
    firm.total_balance = need
    firm.cold_start_share = 1 / sim.PARAMS['FIRM_CAPITAL_MONTHS']
    sim.firms[firm.id] = firm
    return firm


def firm_exit(sim):
    """FIRM_EXIT_MONTHS > 0: firms insolvent (balance <= 0) or idle (no staff, no sales) for that many months in a
    row exit. Reads last month's sales (before reset_amount_sold). Government never exits; Construction only when it
    has no house for sale or under construction."""
    months = sim.PARAMS.get('FIRM_EXIT_MONTHS', 0)
    if months <= 0:
        return
    leaving = []
    for firm in sim.firms.values():
        if firm.sector == 'Government':
            continue
        firm.months_insolvent = firm.months_insolvent + 1 if firm.total_balance <= 0 else 0
        firm.months_idle = firm.months_idle + 1 if not firm.employees and firm.amount_sold == 0 else 0
        if firm.sector == 'Construction' and (firm.houses_for_sale or firm.building):
            continue
        if firm.months_insolvent >= months:
            leaving.append((firm, 'insolvent'))
        elif firm.months_idle >= months:
            leaving.append((firm, 'idle'))
    for firm, reason in leaving:
        exit_firm(sim, firm, reason)


def exit_firm(sim, firm, reason):
    """Removes firm from every structure that holds firms and moves it to sim.firm_grave. Staff become
    unemployed; remaining capital goes in equal parts to the surviving firms of its sector (or, if none, to the
    families); a negative balance is written off and recorded in sim.firm_exit_writeoff."""
    for agent in list(firm.employees.values()):
        agent.firm_id = None
        agent.set_commute(None)
    firm.employees.clear()
    del sim.firms[firm.id]
    if firm.total_balance > 0:
        heirs = [f for f in sim.firms.values() if f.sector == firm.sector]
        if heirs:
            for f in heirs:
                f.total_balance += firm.total_balance / len(heirs)
        else:
            families = [f for f in sim.families.values() if f.members]
            for f in families:
                f.update_balance(firm.total_balance / len(families))
    else:
        sim.firm_exit_writeoff -= firm.total_balance
        sim.ledger['firm_writeoff'] -= firm.total_balance
    firm.total_balance = 0.0
    for product in firm.inventory.values():
        product.quantity = 0
    firm.exit_date = sim.clock.days
    firm.exit_reason = reason
    sim.firm_grave[firm.id] = firm
    # Structures that keep firms across months
    for house in sim.houses.values():
        house._firm_distances.pop(firm.id, None)
    lm = sim.labor_market
    lm.available_postings = [f for f in lm.available_postings if f is not firm]
    for mun, firms in sim.funds.mun_gov_firms.items():
        if firm in firms:
            firms.remove(firm)
