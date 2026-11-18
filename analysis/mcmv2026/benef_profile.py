"""Profile of MCMV beneficiaries, Goiânia d25 and d09, area order (output/mcmv2026) against full targeting
(output/queue_test), seeds 0-9. A beneficiary of 2026-34 is a family tagged 'mcmv' in a January snapshot 2027..2034 and
not tagged in 2026. Profile at the first tagged snapshot: members, age of the oldest member, permanent income per
person, wage per person. Exit: absent from a later snapshot (members died or left; merges counted apart).
Entries: new families owning at entry, MCMV-tagged or not.
usage (repo root, env ps3_2): python analysis/mcmv2026/benef_profile.py [cpus]"""
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOTS = {'area order': 'output/mcmv2026', 'full targeting': 'output/queue_test'}


def one(queue, root, sc, seed):
    d = f'{root}/{sc}/GOIANIA/{seed}'
    flt = [('month', '>=', pd.Timestamp(2026, 1, 1).date())]
    fa = pd.read_parquet(f'{d}/families.parquet', filters=flt)
    ag = pd.read_parquet(f'{d}/agents.parquet', columns=['month', 'id', 'age', 'family_id'], filters=flt)
    fa['year'] = pd.to_datetime(fa.month).dt.year
    ag['year'] = pd.to_datetime(ag.month).dt.year
    old = ag.groupby(['year', 'family_id']).age.max()
    fa['oldest'] = pd.Series(list(zip(fa.year, fa.id))).map(old).values
    fa['pi_pc'] = fa.permanent_income / fa.num_members.clip(lower=1)
    fa['w_pc'] = fa.total_wage / fa.num_members.clip(lower=1)
    pre = set(fa[(fa.year == 2026) & (fa.programme == 'mcmv')].id)
    t = fa[(fa.programme == 'mcmv') & ~fa.id.isin(pre)].sort_values('year')
    first = t.groupby('id').first()
    last_seen = fa.groupby('id').year.max()
    first['exit'] = last_seen.reindex(first.index).values < 2034
    # exit within 3 years of the first tagged snapshot, for beneficiaries first seen up to 2031
    first['exit3'] = (last_seen.reindex(first.index).values < first.year + 3)
    e3 = first[first.year <= 2031]
    # all families of the 2030 snapshot, for the population percentile of beneficiaries
    f30 = fa[fa.year == 2030]
    return dict(queue=queue, scenario=sc, seed=seed, benef=len(first), members=first.num_members.mean(),
                single=(first.num_members == 1).mean(), oldest=first.oldest.mean(),
                over65=(first.oldest >= 65).mean(), zero_wage=(first.total_wage <= 0).mean(),
                pi_pc=first.pi_pc.mean(), w_pc=first.w_pc.mean(), exit_by_2034=first.exit.mean(),
                exit_3y=e3.exit3.mean(), all_members=f30.num_members.mean(),
                all_over65=(f30.id.map(old.xs(2030)) >= 65).mean())


if __name__ == '__main__':
    jobs = [(q, r, sc, s) for q, r in ROOTS.items() for sc in ('d09', 'd25') for s in range(10)]
    res = pd.DataFrame(Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 6)(delayed(one)(*j) for j in jobs))
    res.to_csv('output/queue_test/_analysis/benef_profile.csv', index=False)
    pd.set_option('display.width', 250)
    g = res.groupby(['scenario', 'queue']).mean().drop(columns='seed')
    print(g.T.round(3).to_string())
