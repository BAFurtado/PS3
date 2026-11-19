"""
LABOUR_FLOWS A/B: 'data' (one PME 2010 separation and job-finding rate for every worker) against 'education' (the
rates of the worker's Census education level, layoffs weighted by the level's separation rate), paired by seed.
Settings, saved data and indicators as runner_education_ab.py (ENMU base, 2010-2021, six areas, unemployment by level
against the Census 2010, AP correlations, commute bands). The 'data' side equals runner_education_ab 'off'.

    python runner_flows_ab.py lote CPUS [cities=enmu|A,B] [runs=N] [part=K/N]
    python runner_flows_ab.py comparar

Output: output/flows_ab/<value>/<CITY>/<i>/. Local conf/params.py and conf/run.py are ignored.
"""
import runner_education_ab as X

X.A.OUTPUT = 'output/flows_ab'
X.A.KEY = 'LABOUR_FLOWS'
X.A.LEVELS = ['data', 'education']

if __name__ == '__main__':
    X.A.main()
