"""Tenure flows, Goiânia: area order (output/mcmv2026) against full targeting (output/queue_test), d09 and d25,
seeds 0-9. Between consecutive January snapshots 2026..2034 each family is a stayer, an entry or an exit.
Stayers switch tenure (renter -> owner by MCMV, renter -> owner otherwise, owner -> renter). Entries and exits are
split by tenure and by what happened to their members (agents parquet): an entry whose members were all in another
family the year before is a split (marriage), otherwise it holds a newcomer (immigrant). An exit whose members all
sit in another family the year after is a merge, otherwise its members died or left.
usage (repo root, env ps3_2): python analysis/mcmv2026/tenure_flows.py [cpus]
Writes output/queue_test/_analysis/tenure_flows.csv (one row per run and year pair) and prints d25 - d09 summed over
2026-34 and over 2031-34, seed means, t of the difference between the queues."""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

CITY = 'GOIANIA'
SEEDS = range(10)
ROOTS = {'area order': 'output/mcmv2026', 'full targeting': 'output/queue_test'}
YEARS = range(2025, 2035)
cpus = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def one(queue, root, sc, seed):
    d = f'{root}/{sc}/{CITY}/{seed}'
    flt = [('month', '>=', pd.Timestamp(2025, 1, 1).date())]
    fa = pd.read_parquet(f'{d}/families.parquet', columns=['month', 'id', 'house_rent', 'programme', 'num_members',
                                                           'permanent_income'], filters=flt)
    ag = pd.read_parquet(f'{d}/agents.parquet', columns=['month', 'id', 'family_id'], filters=flt)
    fa['year'] = pd.to_datetime(fa.month).dt.year
    ag['year'] = pd.to_datetime(ag.month).dt.year
    fa['renter'] = fa.house_rent.notna()
    fa['mcmv'] = fa.programme == 'mcmv'
    rows = []
    for y in YEARS[:-1]:
        a = fa[fa.year == y].set_index('id')
        b = fa[fa.year == y + 1].set_index('id')
        ga = ag[ag.year == y].set_index('id').family_id
        gb = ag[ag.year == y + 1].set_index('id').family_id
        stay = a.index.intersection(b.index)
        ent = b.index.difference(a.index)
        ext = a.index.difference(b.index)
        sa, sb = a.loc[stay], b.loc[stay]
        r = dict(queue=queue, scenario=sc, seed=seed, year=y + 1,
                 owners=(~b.renter).sum(), renters=b.renter.sum(),
                 R_O_mcmv=(sa.renter & ~sb.renter & sb.mcmv & ~sa.mcmv).sum(),
                 R_O_other=(sa.renter & ~sb.renter & ~(sb.mcmv & ~sa.mcmv)).sum(),
                 O_R=(~sa.renter & sb.renter).sum())
        # entries: members known the year before (split) or not (newcomer)
        mem_b = gb[gb.isin(ent)]
        known = mem_b.index.isin(ga.index)
        split = pd.Series(known, index=mem_b.values).groupby(level=0).all()
        e = b.loc[ent].assign(split=split.reindex(ent).fillna(False).values)
        for ten, m in (('O', ~e.renter), ('R', e.renter)):
            r[f'in_{ten}_split'] = (m & e.split).sum()
            r[f'in_{ten}_new'] = (m & ~e.split).sum()
        # exits: members found the year after (merge) or not (died, left)
        mem_a = ga[ga.isin(ext)]
        found = mem_a.index.isin(gb.index)
        merge = pd.Series(found, index=mem_a.values).groupby(level=0).all()
        x = a.loc[ext].assign(merged=merge.reindex(ext).fillna(False).values)
        for ten, m in (('O', ~x.renter), ('R', x.renter)):
            r[f'out_{ten}_merge'] = (m & x.merged).sum()
            r[f'out_{ten}_gone'] = (m & ~x.merged).sum()
        r['out_O_mcmv'] = (~x.renter & x.mcmv).sum()
        rows.append(r)
    return rows


if __name__ == '__main__':
    jobs = [(q, root, sc, s) for q, root in ROOTS.items() for sc in ('d09', 'd25') for s in SEEDS]
    res = Parallel(n_jobs=cpus, verbose=2)(delayed(one)(*j) for j in jobs)
    T = pd.DataFrame([r for rr in res for r in rr])
    os.makedirs('output/queue_test/_analysis', exist_ok=True)
    T.to_csv('output/queue_test/_analysis/tenure_flows.csv', index=False)
    flows = [c for c in T.columns if c.startswith(('R_O', 'O_R', 'in_', 'out_'))]
    pd.set_option('display.width', 250)
    for lab, y0 in (('2026-34', 2026), ('2031-34', 2031)):
        S = T[T.year >= y0].groupby(['queue', 'scenario', 'seed'])[flows].sum()
        end = T[T.year == 2034].set_index(['queue', 'scenario', 'seed'])[['owners', 'renters']]
        start = T[T.year == y0 - 1 if y0 > 2026 else T.year == 2026].set_index(['queue', 'scenario', 'seed'])[
            ['owners', 'renters']] if y0 > 2026 else None
        S = S.join(end)
        D = S.xs('d25', level='scenario') - S.xs('d09', level='scenario')
        rows = {}
        for c in D.columns:
            a, b = D.loc['area order', c], D.loc['full targeting', c]
            se = np.sqrt(a.var() / len(a) + b.var() / len(b))
            rows[c] = dict(d09_area=S.xs(('area order', 'd09'), level=['queue', 'scenario'])[c].mean(),
                           area=a.mean(), full=b.mean(), diff=b.mean() - a.mean(), t=(b.mean() - a.mean()) / se)
        print(f'\n== {lab}: flows summed (families), d25 - d09 by queue, seed means; owners/renters = stock Jan 2034')
        print(pd.DataFrame(rows).T.round(1).to_string())
