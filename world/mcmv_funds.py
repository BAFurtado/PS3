from collections import defaultdict


class MCMV:
    def __init__(self, sim):
        self.sim = sim

    def monthly_allocation(self, year):
        """This month's OGU top-up per municipality: `share x municipal GDP / 12`.

        Returns a fresh dictionary of increments. The caller owns the persistent
        pot and adds these to it, so an unspent balance survives into next month.
        """
        allocation = defaultdict(float)

        year = int(year)

        # Default observed values
        if 2010 <= year <= 2016:
            value = 0.25
        elif 2017 <= year <= 2022:
            value = 0.04
        elif 2023 <= year <= 2025:
            value = 0.09
        elif year >= 2026:
            # Read per-run params, not the module global: sensitivity sweeps
            # override sim.PARAMS only (main.py multiple_runs), leaving
            # conf.PARAMS at its defaults.
            params = self.sim.PARAMS
            value = params['OGU_INVESTMENT'][params['FUNDS_AVAILABILITY']]
        else:
            value = 0

        muns = {int(str(mun)[:6]) for mun in self.sim.geo.mun_codes}

        for mun in muns:
            allocation[str(mun)] += (value * self.sim.stats.last_gdp[mun] / 12)

        return allocation