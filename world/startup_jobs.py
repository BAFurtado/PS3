"""Start-up firms and staff by municipality from Census 2010 employees by place of work (STARTUP_FIRMS and
STARTUP_JOBS 'census')"""
import math
from collections import defaultdict

import pandas as pd

FILE = 'input/census_jobs_workplace_2010.csv'


def census_persons(acps):
    """{municipality: Census 2010 employees with or without carteira who work there} for the ACPs, public
    administration and civil servants left out."""
    table = pd.read_csv(FILE, sep=';', dtype={'work_mun': str})
    missing = sorted(set(acps) - set(table.acp))
    if missing:
        raise ValueError("Census 2010 workplace data missing for %s in %s" % (missing, FILE))
    return table[table.acp.isin(acps)].groupby('work_mun').persons.sum().to_dict()


def startup_quotas(firms, acps, jobs):
    """{firm id: staff} for the firms other than Government and own-account pools. A municipality gets its share of
    the Census employees (census_persons) times `jobs`, split equally among its firms and rounded up. Shares are taken
    over the municipalities that have such firms."""
    persons = census_persons(acps)
    by_mun = defaultdict(list)
    for firm in firms:
        if firm.sector != 'Government' and not firm.own_account:
            by_mun[firm.region_id[:7]].append(firm.id)
    total = sum(persons.get(mun, 0.0) for mun in by_mun)
    if total <= 0:
        raise ValueError("STARTUP_JOBS 'census': no Census 2010 employees work where the firms are")
    quotas = dict()
    for mun, ids in by_mun.items():
        staff = math.ceil(persons.get(mun, 0.0) / total * jobs / len(ids))
        for firm_id in ids:
            quotas[firm_id] = staff
    return quotas


def census_firm_quotas(quotas, population, acps):
    """{region id: firms to add}. A municipality whose firms in `quotas` fall short of its share of the Census employees
    (census_persons, shares over the municipalities of `quotas`) times the total of `quotas`, rounded half up, gets
    the missing firms, spread over its regions in proportion to `population` {region id: persons} (equally when they
    have none) by largest remainder."""
    persons = census_persons(acps)
    by_mun = defaultdict(list)
    for rid in sorted(quotas):
        by_mun[rid[:7]].append(rid)
    total_persons = sum(persons.get(mun, 0.0) for mun in by_mun)
    if total_persons <= 0:
        raise ValueError("STARTUP_FIRMS 'census': no Census 2010 employees work in the processed municipalities")
    total = sum(quotas.values())
    result = dict()
    for mun, rids in by_mun.items():
        add = math.floor(persons.get(mun, 0.0) / total_persons * total + 0.5) - sum(quotas[r] for r in rids)
        if add <= 0:
            continue
        weights = [population.get(r, 0) for r in rids]
        if sum(weights) <= 0:
            weights = [1] * len(rids)
        exact = [add * w / sum(weights) for w in weights]
        counts = [math.floor(x) for x in exact]
        order = sorted(range(len(rids)), key=lambda i: exact[i] - counts[i], reverse=True)
        for i in order[:add - sum(counts)]:
            counts[i] += 1
        for rid, n in zip(rids, counts):
            if n:
                result[rid] = n
    return result
