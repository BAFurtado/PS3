"""
HOUSING_BIDDING A/B with transport off: 'off' (buyers one at a time, first house they can pay for) against
'filatova' (buyers bid in rounds, each house to its highest bid, markets/housing.py), paired by seed. Both sides as
the ENMU runs without transport: RENT_LEVEL census_ap, VACANCY_SCOPE region; 2010-2021, seeds SEED + i. Saved:
stats, regional, the houses parquet every January and sales.csv (every sale).

Modes and comparison as runner_rent_ap.py (cities default enmu, the six ENMU areas), plus the price paid over the
listed price of all sales and of those in the top fifth of areas by Census rent, and sales a month.

    python runner_bidding_ab.py lote CPUS [cities=enmu|A,B] [runs=N] [part=K/N]
    python runner_bidding_ab.py comparar

Output: output/bidding_ab/<value>/<CITY>/<i>/. Local conf/params.py and conf/run.py are ignored.
"""
import runner_rent_ap as A

A.OUTPUT = 'output/bidding_ab'
A.KEY = 'HOUSING_BIDDING'
A.LEVELS = ['off', 'filatova']
A.BASE = {'VACANCY_SCOPE': 'region', 'RENT_LEVEL': 'census_ap'}
A.V.R.conf.RUN['SAVE_DATA'] = ['house', 'sales']

if __name__ == '__main__':
    A.main()
