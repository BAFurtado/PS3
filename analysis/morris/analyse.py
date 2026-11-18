"""Morris screen (runner_morris.py): Figure 2 indicators of every run, elementary effects per factor, mu*, sigma.
usage (repo root, env ps3_2): python analysis/morris/analyse.py [runs root, default output/morris2026]
Indicators as analysis/mcmv2026/validation/screen25.py (the Figure 2 source): levels over months 85-120 (2017-19), GDP per
resident over IBGE in 2010 and 2017-19, real GDP growth 2012-19.
Elementary effect of factor j between two points of a trajectory: change of the indicator over the change of the
factor on the unit scale, so an effect is the change from the low to the high end of the range. Effects are taken on
the seed mean of each point and per seed. Each indicator's effects are divided by the width of its band (unemployment:
the city's band; GDP per resident 0.4; real growth 3 pp a year), so 1 = a move across the whole band.
mu* = mean |effect| over trajectories, sigma = sd of effects, noise = mu* of the seed-0 minus seed-1 difference of the
two points' change (the share of an effect that common random numbers do not remove).
Writes <root>/_analysis/morris_runs.csv, morris_effects.csv (trajectory x factor x city x indicator) and
morris_summary.csv (factor x city x indicator: mu_star, sigma, noise, rank)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ['.', 'analysis/mcmv2026/validation']
from analysis.calibration import levels  # noqa: E402

K = 1000.0
HERE = 'analysis/mcmv2026/validation'
LEVELS = levels.HOUSING + ['nontradable_inflation', 'unemployment']
HELD = ['gdp_pc_y1', 'gdp_pc_1719', 'real_gdp_12_19']
INDICATORS = LEVELS + HELD


def run_metrics(d, acp, gdp):
    import dashboard as D
    df = D.load('.', d)
    if len(df) < 120:
        return None
    m = levels.level_moments(df.iloc[:120], 36)
    p = df.price_level / df.price_level.iloc[0]
    gdp_pc = df.gdp_level / df['pop'] * K
    y = pd.to_datetime(df.month).dt.year
    real = (df.gdp_level / df.price_level).groupby(y).mean()
    return dict(**{k: m[k] for k in LEVELS},
                gdp_pc_y1=gdp_pc.iloc[:12].mean() / gdp.loc[acp, 'gdp_pc'],
                gdp_pc_1719=(gdp_pc / p).iloc[84:120].mean() / gdp.loc[acp, 'gdp_pc'],
                real_gdp_12_19=((real[2019] / real[2012]) ** (1 / 7) - 1) * 100)


def widths(cities):
    bands = levels.load_targets(cities)
    gb = pd.read_csv(f'{HERE}/gate_bounds25.csv').set_index('acp')
    w = {}
    for c in cities:
        w[c] = {k: bands[c][k][1] - bands[c][k][0] for k in levels.HOUSING + ['nontradable_inflation']}
        w[c]['unemployment'] = gb.loc[c, 'pnad_max'] - min(gb.loc[c, 'census_u'], gb.loc[c, 'pnad_min'])
        w[c].update(gdp_pc_y1=.4, gdp_pc_1719=.4, real_gdp_12_19=3.)
    return w


def effects(design, y, factors):
    """Elementary effects. design: point, trajectory, step, u_<factor>; y: indicator values indexed by point.
    Returns {(trajectory, factor): effect}."""
    out = {}
    for t, g in design.sort_values('step').groupby('trajectory'):
        U = g[['u_' + f for f in factors]].values
        pts = g.point.values
        for j in range(1, len(g)):
            du = U[j] - U[j - 1]
            i = int(np.flatnonzero(np.abs(du) > 1e-9)[0])
            if pts[j] in y.index and pts[j - 1] in y.index:
                out[t, factors[i]] = (y[pts[j]] - y[pts[j - 1]]) / du[i]
    return out


def summarise(E):
    g = E.groupby(['city', 'indicator', 'factor'])
    S = pd.DataFrame({'mu_star': g.effect.apply(lambda e: e.abs().mean()), 'mu': g.effect.mean(),
                      'sigma': g.effect.std(), 'noise': g.noise.apply(lambda e: e.abs().mean()),
                      'n': g.effect.count()}).reset_index()
    S['rank'] = S.groupby(['city', 'indicator']).mu_star.rank(ascending=False).astype(int)
    return S.sort_values(['city', 'indicator', 'rank'])


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'output/morris2026'
    out = os.path.join(root, '_analysis')
    os.makedirs(out, exist_ok=True)
    problem = json.load(open('analysis/morris/problem.json'))
    factors = list(problem['factors'])
    design = pd.read_csv('analysis/morris/design.csv')
    gdp = pd.read_csv(f'{HERE}/apu_va25.csv').set_index('acp')

    rows = []
    for f in sorted(glob.glob(os.path.join(root, '*', '*', '*', 'DONE'))):
        d = os.path.dirname(f)
        city, point, i = d.split(os.sep)[-3:]
        m = run_metrics(d, city, gdp)
        if m is None:
            print('incomplete', d)
            continue
        rows.append(dict(city=city, point=point, seed=int(i), **m))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, 'morris_runs.csv'), index=False, float_format='%.6g')
    W = widths(sorted(R.city.unique()))

    E = []
    for city in sorted(R.city.unique()):
        for ind in INDICATORS:
            r = R[R.city == city].pivot(index='point', columns='seed', values=ind)
            full = r.dropna()
            mean = effects(design, full.mean(axis=1), factors)
            per = [effects(design, full[s], factors) for s in full.columns]
            for key, e in mean.items():
                noise = (per[0][key] - per[1][key]) / 2 if len(per) > 1 else np.nan
                E.append(dict(city=city, indicator=ind, trajectory=key[0], factor=key[1], effect=e / W[city][ind],
                              noise=noise / W[city][ind]))
    E = pd.DataFrame(E)
    E.to_csv(os.path.join(out, 'morris_effects.csv'), index=False, float_format='%.6g')
    S = summarise(E)
    S.to_csv(os.path.join(out, 'morris_summary.csv'), index=False, float_format='%.4g')

    pd.set_option('display.width', 220)
    print(R.groupby('city').agg(points=('point', 'nunique'), runs=('point', 'count')).to_string())
    for (city, ind), g in S.groupby(['city', 'indicator']):
        print('\n== {} {} (top 8; effect in band widths)'.format(city, ind))
        print(g.head(8)[['factor', 'mu_star', 'mu', 'sigma', 'noise', 'n']].round(3).to_string(index=False))
