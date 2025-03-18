import conf
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger('stats')

if conf.RUN['PRINT_STATISTICS_AND_RESULTS_DURING_PROCESS']:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.ERROR)

head_rate = dict()
j = 15
for i in range(13):
    head_rate[f'{j + i}-{j + i + 4}'] = int
    j += 4


class Statistics(object):
    """
    The statistics class contains a bundle of functions together
    The functions include average price of the firms, regional GDP - based on FIRMS' revenues, GDP per
    capita, unemployment, families' wealth, GINI, regional GINI and commuting information.
    """

    def __init__(self, params):
        self.previous_month_price = 0
        self.global_unemployment_rate = .086
        self.vacancy_rate = params['HOUSE_VACANCY']

    def calculate_firms_metrics(self, firms):
        """Compute median firms values in one pass."""
        n_firms = len(firms)
        firm_balances = np.zeros(n_firms)
        firm_wages = np.zeros(n_firms)
        firm_eco_eff = np.zeros(n_firms)
        firm_emissions = np.zeros(n_firms)
        firm_stocks = np.zeros(n_firms)
        firm_workers = np.zeros(n_firms)
        firm_profits = np.zeros(n_firms)

        for i, firm in enumerate(firms.values()):
            firm_balances[i] = firm.total_balance
            firm_wages[i] = firm.wages_paid
            firm_eco_eff[i] = firm.env_efficiency
            firm_emissions[i] = firm.last_emissions
            firm_stocks[i] = firm.get_total_quantity()
            firm_workers[i] = firm.num_employees
            firm_profits[i] = firm.profit

        results = {
            "median_wealth": np.median(firm_balances) if firm_balances.size > 0 else 0,
            "median_wages": np.median(firm_wages) if firm_wages.size > 0 else 0,
            "eco_efficiency": np.median(firm_eco_eff) if firm_eco_eff.size > 0 else 0,
            "emissions": np.median(firm_emissions) if firm_emissions.size > 0 else 0,
            "median_stock": np.median(firm_stocks) if firm_stocks.size > 0 else 0,
            "workers": np.median(firm_workers) if firm_workers.size > 0 else 0,
            "aggregate_profits": np.sum(firm_profits) if firm_profits.size > 0 else 0,
        }
        logger.info(f"Firm stats - Median wealth: {results['median_wealth']:.2f}, "
                    f"Median wages: {results['median_wages']:.2f}, "
                    f"Eco Efficiency: {results['eco_efficiency']:.2f}, "
                    f"Emissions: {results['emissions']:.2f}, "
                    f"Median stock: {results['median_stock']:.2f}, "
                    f"Median workers: {results['workers']:.2f}, "
                    f"Aggregate profits: {results['aggregate_profits']:.2f}"
        )

        return results

    def update_price(self, firms, mid_simulation_calculus=False):
        """Compute average price and inflation"""
        prices = [item.price for firm in firms.values() for item in firm.inventory.values()
                  if item.quantity > 0 and firm.num_employees > 0]

        average_price = np.mean(prices) if prices else 0

        # Use saved price to calculate inflation
        inflation = ((average_price - self.previous_month_price) / self.previous_month_price
                     if self.previous_month_price else 0)

        # Save current prices to be used next month
        if not mid_simulation_calculus:
            self.previous_month_price = average_price
            logger.info(f'Price average: {average_price:.3f}, Monthly inflation: {inflation:.3f}')
        return average_price, inflation

    def calculate_gdp_and_eco_efficiency(self, firms, regions):
        """Calculate GDP and Eco-Efficiency for all regions using NumPy arrays for maximum efficiency."""

        total_gdp = 0
        previous_total_gdp = sum(region.gdp for region in regions.values())  # Store previous GDP
        n_firms = len(firms)

        # Preallocate arrays for firm revenues and eco-efficiencies, mapped to region IDs
        firm_revenues = np.zeros(n_firms)
        firm_eco_efficiencies = np.zeros(n_firms)
        firm_region_ids = np.zeros(n_firms, dtype=int)

        # SINGLE loop through firms to populate arrays
        for i, firm in enumerate(firms.values()):
            firm_revenues[i] = firm.revenue
            firm_eco_efficiencies[i] = firm.env_efficiency
            firm_region_ids[i] = firm.region_id

        # Compute metrics for each region
        for region in regions.values():
            mask = firm_region_ids == region.id  # Efficient NumPy filtering

            region.gdp = np.sum(firm_revenues[mask]) if np.any(mask) else 0
            region.avg_eco_eff = np.mean(firm_eco_efficiencies[mask]) if np.any(mask) else 0
            total_gdp += region.gdp

        # Compute GDP growth
        gdp_growth = ((total_gdp - previous_total_gdp) / total_gdp) * 100 if total_gdp != 0 else 1
        logger.info(f'GDP index variation: {gdp_growth:.2f}%')

        return total_gdp, gdp_growth

    def calculate_avg_regional_house_price(self, regional_families):
        return np.average([f.house.price for f in regional_families if f.num_members > 0])

    def calculate_house_metrics(self, houses):
        """Compute various house-level metrics efficiently."""
        n_houses = len(houses)
        # Initialize NumPy arrays
        house_prices = np.zeros(n_houses)
        rent_prices = np.zeros(n_houses)
        has_rent_data = np.zeros(n_houses, dtype=bool)
        is_vacant = np.zeros(n_houses, dtype=bool)

        # Fill arrays
        for i, house in enumerate(houses.values()):
            house_prices[i] = house.price
            if house.rent_data is not None:
                rent_prices[i] = house.rent_data[0]
                has_rent_data[i] = True
            is_vacant[i] = house.family_id is None

        # Compute metrics using NumPy
        avg_house_price = np.mean(house_prices)
        avg_rent_price = np.mean(rent_prices[has_rent_data]) if np.any(has_rent_data) else 0
        vacancy_rate = np.sum(is_vacant) / n_houses if n_houses > 0 else 0
        self.vacancy_rate = vacancy_rate

        # Logging (if enabled)
        logger.info(f'Vacant houses {np.sum(is_vacant):,.0f}')
        logger.info(f'Total houses {n_houses:,.0f}')

        return {
            "average_house_price": avg_house_price,
            "average_rent_price": avg_rent_price,
            "vacancy_rate": vacancy_rate,
        }

    def update_GDP_capita(self, firms, mun_id, mun_pop):
        dummy_gdp = np.sum([firms[firm].revenue for firm in firms.keys()
                            if firms[firm].region_id[:7] == mun_id])
        if mun_pop > 0:
            dummy_gdp_capita = dummy_gdp / mun_pop
        else:
            dummy_gdp_capita = dummy_gdp
        return dummy_gdp_capita

    def update_unemployment(self, agents, global_u=False,log=False):
        employable = [m for m in agents if 16 < m.age < 70]
        temp = len([m for m in employable if m.firm_id is None]) / len(employable) if employable else 0
        if log:
            logger.info(f'Unemployment rate: {temp * 100:.2f}')
        if global_u:
            self.global_unemployment_rate = temp
        return temp

    def calculate_head_rate(self, families):
        for f in families:
            head_agent = 0
            head_rate

    def calculate_families_metrics(self, families):
        """Compute various family-level metrics efficiently."""
        n_families = len(families)
        # Initialize NumPy arrays
        renting = np.zeros(n_families, dtype=bool)
        permanent_income = np.zeros(n_families)
        rent_ratio = np.zeros(n_families)
        has_rent_voucher = np.zeros(n_families, dtype=bool)
        savings = np.zeros(n_families)
        wages = np.zeros(n_families)
        utility = np.zeros(n_families)
        rent_default = np.zeros(n_families, dtype=bool)
        num_members = np.zeros(n_families, dtype=int)

        # Fill arrays
        for i, family in enumerate(families.values()):
            renting[i] = family.is_renting
            permanent_income[i] = family.get_permanent_income()
            savings[i] = family.savings
            wages[i] = family.total_wage()
            utility[i] = family.average_utility
            rent_default[i] = family.rent_default == 1 and family.is_renting
            num_members[i] = family.num_members
            head_family = max(family.members.values(), key=lambda x: x.last_wage)
            head_family.set_head_family()
            if family.is_renting:
                has_rent_voucher[i] = family.rent_voucher
                rent_ratio[i] = family.house.rent_data[0] / (permanent_income[i] if permanent_income[i] > 0 else 1)

        # Compute metrics using NumPy
        total_renting = np.sum(renting)
        affordable = np.sum((renting & ~has_rent_voucher & (permanent_income > 0) & (rent_ratio < 0.3)))

        affordability_ratio = affordable / total_renting if total_renting > 0 else 0
        median_wealth = np.median(permanent_income)
        median_affordability = np.median(rent_ratio[renting]) if total_renting > 0 else 0
        median_wages = np.median(wages)
        total_savings = np.sum(savings)
        rent_default_ratio = np.sum(rent_default) / total_renting if total_renting > 0 else 0
        zero_consumption_ratio = np.sum(utility == 0) / n_families if n_families > 0 else 0
        avg_utility = np.average(utility[num_members > 0])

        # GINI calculation
        sorted_income = np.sort(permanent_income + 1e-7)  # Avoid division by zero
        n = sorted_income.size
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_income) / (n * np.sum(sorted_income))) if n > 0 else 0

        return {
            "affordability_ratio": affordability_ratio,
            "median_wealth": median_wealth,
            "median_affordability": median_affordability,
            "median_wages": median_wages,
            "total_savings": total_savings,
            "rent_default_ratio": rent_default_ratio,
            "zero_consumption_ratio": zero_consumption_ratio,
            "avg_utility": avg_utility,
            "gini": gini,
        }

    def calculate_regional_gini(self, families):
        n_families = len(families)
        permanent_income = np.zeros(n_families)
        for i, family in enumerate(families):
            permanent_income[i] = family.get_permanent_income()
        # GINI calculation
        sorted_income = np.sort(permanent_income + 1e-7)  # Avoid division by zero
        n = sorted_income.size
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_income) / (n * np.sum(sorted_income))) if n > 0 else 0
        return gini

    def update_commuting(self, families):
        """Total commuting distance"""
        dummy_total = 0.
        for family in families:
            for member in family.members.values():
                if member.is_employed:
                    dummy_total += member.distance
        return dummy_total

    def average_qli(self, regions):
        # group by municipality
        mun_regions = defaultdict(list)
        for id, region in regions.items():
            mun_code = id[:7]
            mun_regions[mun_code].append(region.index)

        average = 0
        for indices in mun_regions.values():
            mun_qli = sum(indices) / len(indices)
            average += mun_qli
        return average / len(mun_regions)
