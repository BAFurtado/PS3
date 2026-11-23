"""
Validation of the ENMU base runs (runner_enmu.py, scenario base) against the Census, below the metro average.

    python analysis/enmu/validacao.py [ROOT]          ROOT default output/enmu2026_rede

Model side: the employees of each base run (employed agents with commute minutes above 0; own-account pool members
have 0 and are left out), January snapshots, mean over seeds of the shares. Mode: car or motorcycle owners commute by
car minutes, the others by transit minutes. Education level of the agent's years of study (world/own_account.level).
The network is ENMU's Rede Base (2054, morning peak) in every year.

1. Census 2022 by municipality and mode (input/commute_mode_2022.csv, Sidra 10330: employed persons working away
   from home, own-account included), model January 2022 and 2023. Bands up to 30, 31-60, 61-120, over 120 minutes,
   shares of each mode's commuters. Private: car, motorcycle, taxi, motorcycle taxi. Transit: bus, BRT, van, rail.
2. Census 2010 by AP and education level (input/commute_ap_education_2010.csv, employees who return home daily,
   walkers included), model January 2011. Share over 60 minutes and mean minutes at class midpoints (3, 18, 45, 90,
   150), so model minutes are put in the same classes first. The Census has no mode in 2010.
3. Population by municipality (input/census_population_2010_2022.csv): growth 2010-2022 and the 2022 share of the
   metro, model January 2010 and 2022.
4. Where employees work (input/commute_workplace_2010.csv, Census 2010 employees who return home daily), model
   January 2011: share of each municipality's employees working in their own municipality, and of the periphery's
   employees working in the core.

Across municipalities or APs: Spearman correlation of model and Census, and the population-weighted mean absolute
difference. Core: the metro's most populous municipality.

Output in ROOT/validacao/: municipio_modo_2022.csv, ap_nivel_2010.csv, populacao_municipio.csv,
local_trabalho_2010.csv, resumo.txt.
"""
import datetime
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'output/enmu2026_rede'
OUT = os.path.join(ROOT, 'validacao')
SCENARIO = 'base'
YEARS_2022 = (2022, 2023)
YEAR_2010 = 2011
MIN_COMMUTERS = 30  # model employees per seed, below which a municipality or AP-level cell is left out of the summary

BANDS_2022 = {'Até cinco minutos': '0-30', 'De seis minutos até quinze minutos': '0-30',
              'Mais de quinze minutos até meia hora': '0-30', 'Mais de meia hora até uma hora': '31-60',
              'Mais de uma hora até duas horas': '61-120', 'Mais de duas horas até quatro horas': '120+',
              'Mais de quatro horas': '120+'}
BAND_ORDER = ['0-30', '31-60', '61-120', '120+']
MODES_2022 = {'car': 'private', 'motorcycle': 'private', 'taxi': 'private', 'motorcycle_taxi': 'private',
              'bus': 'transit', 'brt': 'transit', 'van': 'transit', 'rail': 'transit'}
MIDPOINT = {1: 3, 2: 18, 3: 45, 4: 90, 5: 150}


def band_2022(minutes):
    return pd.cut(minutes, [-np.inf, 30, 60, 120, np.inf], labels=BAND_ORDER)


def class_2010(minutes):
    return pd.cut(minutes, [-np.inf, 5, 30, 60, 120, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)


def level(q):
    return np.where(q < 4, 1, np.where(q < 9, 2, np.where(q < 12, 3, 4)))


def base_runs(root):
    out = {}
    for done in glob.glob(os.path.join(root, SCENARIO, '*', '*', 'DONE')):
        d = os.path.dirname(done)
        city, i = d.split(os.sep)[-2:]
        out.setdefault(city, []).append(d)
    return out


def snapshot(d, years):
    ag = pd.read_parquet(os.path.join(d, 'agents.parquet'),
                         columns=['month', 'region_id', 'firm_id', 'distance', 'has_car', 'qualification'])
    ag = ag[ag.month.isin([datetime.date(y, 1, 1) for y in years])]
    return ag


def spearman(a, b):
    ok = a.notna() & b.notna()
    return a[ok].corr(b[ok], method='spearman') if ok.sum() > 2 else np.nan


def wmae(a, b, w):
    ok = a.notna() & b.notna() & w.notna()
    return np.average((a[ok] - b[ok]).abs(), weights=w[ok]) if ok.any() else np.nan


def model_2022(dirs):
    """Shares by municipality, mode and band, mean over seeds; and employees per seed"""
    rows = []
    for k, d in enumerate(dirs):
        ag = snapshot(d, YEARS_2022)
        e = ag[ag.firm_id.notna() & (ag.distance > 0)].copy()
        e['cod_mun'] = e.region_id.str[:7].astype(int)
        e['mode'] = np.where(e.has_car, 'private', 'transit')
        e['band'] = band_2022(e.distance)
        n = e.groupby(['cod_mun', 'mode', 'band'], observed=False).size().rename('n').reset_index()
        n['seed'] = k
        rows.append(n)
    n = pd.concat(rows)
    n['n'] = n.n / len(YEARS_2022)
    tot = n.groupby(['seed', 'cod_mun', 'mode']).n.transform('sum')
    n['share'] = n.n / tot
    m = n.groupby(['cod_mun', 'mode', 'band'], observed=False).agg(model_share=('share', 'mean'),
                                                                    model_n=('n', 'mean')).reset_index()
    return m


def census_2022(acp):
    t = pd.read_csv('input/commute_mode_2022.csv', sep=';')
    t = t[(t.acp == acp) & t['mode'].isin(MODES_2022)]
    t = t.assign(mode=t['mode'].map(MODES_2022), band=t.band.map(BANDS_2022))
    t = t.groupby(['cod_mun', 'mode', 'band']).persons.sum().reset_index()
    t['census_share'] = t.persons / t.groupby(['cod_mun', 'mode']).persons.transform('sum')
    return t.rename(columns={'persons': 'census_n'})


def model_2010(dirs):
    rows = []
    for k, d in enumerate(dirs):
        ag = snapshot(d, [YEAR_2010])
        e = ag[ag.firm_id.notna() & (ag.distance > 0)].copy()
        e['level'] = level(e.qualification.to_numpy())
        e['commute_class'] = class_2010(e.distance)
        n = e.groupby(['region_id', 'level', 'commute_class']).size().rename('n').reset_index()
        n['seed'] = k
        rows.append(n)
    return pd.concat(rows).rename(columns={'region_id': 'ap'})


def class_summary(t, keys, weight):
    """Share over 60 minutes and mean midpoint minutes by `keys`"""
    t = t.assign(long=t[weight] * (t.commute_class >= 4), mid=t[weight] * t.commute_class.map(MIDPOINT))
    g = t.groupby(keys)[[weight, 'long', 'mid']].sum()
    return pd.DataFrame({'n': g[weight], 'over60': g['long'] / g[weight], 'minutes': g['mid'] / g[weight]})


def workplace(dirs):
    """Model employees by municipality of residence and of work, January YEAR_2010, mean over seeds"""
    rows = []
    for d in dirs:
        ag = snapshot(d, [YEAR_2010])
        e = ag[ag.firm_id.notna() & (ag.distance > 0)]
        f = pd.concat([pd.read_parquet(os.path.join(d, n + '.parquet'), columns=['month', 'firm_id', 'region_id'])
                       for n in ('firms', 'construction')])
        f = f[f.month == datetime.date(YEAR_2010, 1, 1)].drop_duplicates('firm_id').set_index('firm_id').region_id
        e = e.assign(cod_mun=e.region_id.str[:7].astype(int), work_mun=e.firm_id.map(f).str[:7].astype(float))
        rows.append(e.dropna(subset=['work_mun']).groupby(['cod_mun', 'work_mun']).size())
    return pd.concat(rows, axis=1).fillna(0).mean(axis=1).rename('model').reset_index().astype({'work_mun': int})


def population(acp, dirs):
    c = pd.read_csv('input/census_population_2010_2022.csv', sep=';')
    codes = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    c = c[c.cod_mun.isin(codes[codes.ACPs == acp].cod_mun)].set_index('cod_mun')
    rows = []
    for k, d in enumerate(dirs):
        ag = pd.read_parquet(os.path.join(d, 'agents.parquet'), columns=['month', 'region_id'])
        ag = ag[ag.month.isin([datetime.date(2010, 1, 1), datetime.date(2022, 1, 1)])]
        n = ag.assign(cod_mun=ag.region_id.str[:7].astype(int), year=pd.to_datetime(ag.month).dt.year) \
            .groupby(['cod_mun', 'year']).size().unstack('year')
        n['seed'] = k
        rows.append(n.reset_index())
    m = pd.concat(rows).groupby('cod_mun')[[2010, 2022]].mean()
    t = c.join(m, how='inner')
    t['census_growth'] = t.pop_2022 / t.pop_2010 - 1
    t['model_growth'] = t[2022] / t[2010] - 1
    t['census_share_2022'] = t.pop_2022 / t.pop_2022.sum()
    t['model_share_2022'] = t[2022] / t[2022].sum()
    return t.rename(columns={2010: 'model_2010', 2022: 'model_2022'}).reset_index()


def main():
    runs = base_runs(ROOT)
    if not runs:
        sys.exit('No base runs in ' + ROOT)
    os.makedirs(OUT, exist_ok=True)
    c2010 = pd.read_csv('input/commute_ap_education_2010.csv', sep=';', dtype={'ap': str})
    t22, t10, tpop, twork, text = [], [], [], [], []
    cwork = pd.read_csv('input/commute_workplace_2010.csv', sep=';')
    for acp in sorted(runs):
        dirs = sorted(runs[acp])
        pop = population(acp, dirs)
        core = int(pop.sort_values('pop_2022').cod_mun.iloc[-1])
        pop['acp'] = acp
        tpop.append(pop)
        text.append('== {} ({} seeds, core {})'.format(acp, len(dirs), core))

        # 1. Census 2022 by municipality and mode
        m = model_2022(dirs).merge(census_2022(acp), on=['cod_mun', 'mode', 'band'], how='outer')
        m['acp'] = acp
        t22.append(m)
        text.append('1. Share of commuters by band, Census 2022 vs model {}-{}, by mode'.format(*YEARS_2022))
        for mode in ('private', 'transit'):
            g = m[m['mode'] == mode]
            pooled = g.groupby('band', observed=False)[['census_n', 'model_n']].sum()
            pooled = pooled / pooled.sum()
            core_rest = g.assign(part=np.where(g.cod_mun == core, 'core', 'rest')) \
                .groupby(['part', 'band'], observed=False)[['census_n', 'model_n']].sum()
            core_rest = core_rest / core_rest.groupby(level=0).transform('sum')
            text.append('  {}: metro {}'.format(mode, '  '.join(
                '{} {:.2f}/{:.2f}'.format(b, pooled.loc[b, 'census_n'], pooled.loc[b, 'model_n'])
                for b in BAND_ORDER if b in pooled.index)))
            for part in ('core', 'rest'):
                if part in core_rest.index.get_level_values(0):
                    cr = core_rest.loc[part]
                    text.append('    {:4s} {}'.format(part, '  '.join(
                        '{} {:.2f}/{:.2f}'.format(b, cr.loc[b, 'census_n'], cr.loc[b, 'model_n'])
                        for b in BAND_ORDER if b in cr.index)))
            size = g.groupby('cod_mun').model_n.sum()
            big = size[size >= MIN_COMMUTERS].index
            if len(big) > 2:
                for b in ('0-30', '61-120'):
                    x = g[(g.band == b) & g.cod_mun.isin(big)].set_index('cod_mun')
                    text.append('    across {} municipalities, share {}: Spearman {:.2f}, weighted MAE {:.3f}'.format(
                        len(x), b, spearman(x.census_share, x.model_share),
                        wmae(x.census_share, x.model_share, x.census_n)))
        text.append('  (census/model; rest = the other municipalities pooled)')

        # 2. Census 2010 by AP and education level
        c = c2010[c2010.acp == acp]
        mo = model_2010(dirs)
        mo = mo[mo.ap.isin(c.ap.unique())]
        seeds = mo.seed.nunique()
        mo = mo.groupby(['ap', 'level', 'commute_class']).n.sum().div(seeds).reset_index()
        ca, ma = class_summary(c, ['ap'], 'persons'), class_summary(mo, ['ap'], 'n')
        ap = ca.join(ma, lsuffix='_census', rsuffix='_model', how='outer')
        cl, ml = class_summary(c, ['level'], 'persons'), class_summary(mo, ['level'], 'n')
        cal, mal = class_summary(c, ['ap', 'level'], 'persons'), class_summary(mo, ['ap', 'level'], 'n')
        apl = cal.join(mal, lsuffix='_census', rsuffix='_model', how='outer').reset_index()
        apl['acp'] = acp
        t10.append(apl)
        text.append('2. Census 2010 employees vs model {}, share over 60 min (mean minutes at class midpoints)'.format(
            YEAR_2010))
        text.append('  metro: census {:.3f} ({:.1f} min), model {:.3f} ({:.1f} min)'.format(
            *class_summary(c.assign(k=1), ['k'], 'persons').iloc[0][['over60', 'minutes']],
            *class_summary(mo.assign(k=1), ['k'], 'n').iloc[0][['over60', 'minutes']]))
        text.append('  by level 1-4: census {}  model {}'.format(
            ' '.join('{:.3f}'.format(v) for v in cl.over60), ' '.join('{:.3f}'.format(v) for v in ml.over60)))
        big = ap[ap.n_model >= MIN_COMMUTERS]
        text.append('  across {} APs: Spearman over60 {:.2f}, minutes {:.2f}, weighted MAE over60 {:.3f}'.format(
            len(big), spearman(big.over60_census, big.over60_model), spearman(big.minutes_census, big.minutes_model),
            wmae(big.over60_census, big.over60_model, big.n_census)))
        # Within-AP education gap: level 1 minus level 3-4 share over 60 min, APs with both in the model
        gap = apl.assign(grp=np.where(apl.level == 1, 'low', np.where(apl.level >= 3, 'high', 'mid')))
        gg = gap[gap.grp != 'mid'].groupby(['ap', 'grp'])[['n_census', 'n_model']].sum()
        for src in ('census', 'model'):
            gap['long_' + src] = gap['over60_' + src] * gap['n_' + src]
        lg = gap[gap.grp != 'mid'].groupby(['ap', 'grp'])[['long_census', 'long_model']].sum()
        sh = (lg / gg.values).unstack('grp')
        ok = (gg['n_model'].unstack('grp') >= 10).all(axis=1)
        sh = sh[ok]
        d_c = sh[('long_census', 'low')] - sh[('long_census', 'high')]
        d_m = sh[('long_model', 'low')] - sh[('long_model', 'high')]
        text.append('  within-AP gap, level 1 minus levels 3-4, share over 60 min ({} APs): census mean {:+.3f}, '
                    'model {:+.3f}, Spearman {:.2f}'.format(len(sh), d_c.mean(), d_m.mean(), spearman(d_c, d_m)))

        # 3. Population by municipality
        big = pop[pop.model_2010 >= 100]
        text.append('3. Population by municipality ({} with >= 100 agents in 2010), model vs Census'.format(len(big)))
        text.append('  growth 2010-2022: metro census {:+.3f}, model {:+.3f}; core census {:+.3f}, model {:+.3f}'.format(
            pop.pop_2022.sum() / pop.pop_2010.sum() - 1, pop.model_2022.sum() / pop.model_2010.sum() - 1,
            *pop.loc[pop.cod_mun == core, ['census_growth', 'model_growth']].iloc[0]))
        rest = pop[pop.cod_mun != core]
        if len(rest):
            text.append('  rest pooled: census {:+.3f}, model {:+.3f}'.format(
                rest.pop_2022.sum() / rest.pop_2010.sum() - 1, rest.model_2022.sum() / rest.model_2010.sum() - 1))
        if len(big) > 2:
            text.append('  across municipalities: Spearman growth {:.2f}, share 2022 {:.2f}'.format(
                spearman(big.census_growth, big.model_growth), spearman(big.census_share_2022, big.model_share_2022)))
        # 4. Where employees work
        cw = cwork[cwork.acp == acp].rename(columns={'persons': 'census'})
        wk = workplace(dirs).merge(cw[['cod_mun', 'work_mun', 'census']], on=['cod_mun', 'work_mun'], how='outer') \
            .fillna({'model': 0, 'census': 0})
        wk['acp'] = acp
        twork.append(wk)
        tot = wk.groupby('cod_mun')[['model', 'census']].sum()
        own = wk[wk.cod_mun == wk.work_mun].set_index('cod_mun')[['model', 'census']].reindex(tot.index).fillna(0)
        share = own / tot
        rest = wk[wk.cod_mun != core]
        text.append('4. Employees by place of work, Census 2010 vs model {}'.format(YEAR_2010))
        if len(rest):
            r_tot = rest[['model', 'census']].sum()
            r_core = rest[rest.work_mun == core][['model', 'census']].sum()
            r_own = rest[rest.cod_mun == rest.work_mun][['model', 'census']].sum()
            text.append('  periphery employees working in the core: census {:.3f}, model {:.3f}; in their own '
                        'municipality: census {:.3f}, model {:.3f}'.format(
                            r_core.census / r_tot.census, r_core.model / r_tot.model,
                            r_own.census / r_tot.census, r_own.model / r_tot.model))
            big = tot[(tot.model >= 100) & (tot.index != core)].index
            text.append('  own-municipality share by periphery municipality ({} with >= 100 model employees): {}'.format(
                len(big), ', '.join('{} {:.2f}/{:.2f}'.format(m, share.loc[m, 'census'], share.loc[m, 'model'])
                                    for m in big)))
        text.append('  core employees working in the core: census {:.3f}, model {:.3f}'.format(
            share.loc[core, 'census'], share.loc[core, 'model']))
        jobs = wk.groupby('work_mun')[['model', 'census']].sum()
        jobs = jobs / jobs.sum()
        res = tot / tot.sum()
        text.append('  core share of the metro: employee jobs census {:.3f}, model {:.3f}; employee residents census '
                    '{:.3f}, model {:.3f}'.format(jobs.loc[core, 'census'] if core in jobs.index else np.nan,
                                                  jobs.loc[core, 'model'] if core in jobs.index else np.nan,
                                                  res.loc[core, 'census'], res.loc[core, 'model']))
        text.append('')
    pd.concat(twork).to_csv(os.path.join(OUT, 'local_trabalho_2010.csv'), sep=';', index=False)
    pd.concat(t22).to_csv(os.path.join(OUT, 'municipio_modo_2022.csv'), sep=';', index=False)
    pd.concat(t10).to_csv(os.path.join(OUT, 'ap_nivel_2010.csv'), sep=';', index=False)
    pd.concat(tpop).to_csv(os.path.join(OUT, 'populacao_municipio.csv'), sep=';', index=False)
    text = '\n'.join(text)
    print(text)
    with open(os.path.join(OUT, 'resumo.txt'), 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
