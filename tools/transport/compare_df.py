"""
Our DF skims against the partner (ENMU) matrix, travel_times_areapond_DF.parquet.

    python -m tools.transport.compare_df

What PS3 uses is the ranking of origins as seen from each destination (the commute term is min-max normalised
within a firm's candidate sample, markets/labor.py), so beyond pooled level/delta agreement the script reports the
per-destination Spearman correlation across origins, for levels and for Nec - Base deltas.
Scope 'df' = the 51 APs inside the DF (the only area our bus data covers); 'acp' = the 84 APs of ACP BRASILIA (what the model loads); 'all' = the partner's 112 codes.
Writes input/bndes/skims/compare_df.csv (pair level) and prints the summary.
"""
import numpy as np
import pandas as pd

from tools.transport.skim import OUT, PARTNER


def load(states=('current', 'base', 'nec')):
    p = pd.read_parquet(PARTNER).rename(columns={'code_weighting_orig': 'ap_orig', 'code_weighting_dest': 'ap_dest',
                                                 'TempoRedeBase': 'enmu_base', 'TempoRedeNec': 'enmu_nec'})
    p = p.set_index(['ap_orig', 'ap_dest'])
    for s in states:
        ours = pd.read_parquet('%s/ap_%s.parquet' % (OUT, s)).set_index(['ap_orig', 'ap_dest'])
        p['ours_' + s] = ours.minutes
        p['unreached_' + s] = ours.unreached
    return p.reset_index()


def acp_codes():
    acp = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    return set(acp[acp.iloc[:, 0] == 'BRASILIA'].iloc[:, 1].astype(str))


def per_dest_spearman(d, a, b):
    r = d.groupby('ap_dest').apply(lambda g: g[a].corr(g[b], method='spearman') if g[[a, b]].dropna().shape[0] > 5
                                   else np.nan, include_groups=False)
    return r.median(), r.quantile(.1), r.quantile(.9)


def summary(d, label):
    off = d[d.ap_orig != d.ap_dest]
    rows = []
    for ours in ('ours_current', 'ours_base'):
        x = off[[ours, 'enmu_base']].dropna()
        rows.append(dict(scope=label, compare='level %s vs enmu_base' % ours, n=len(x),
                         pearson=x[ours].corr(x.enmu_base), spearman=x[ours].corr(x.enmu_base, method='spearman'),
                         mean_ours=x[ours].mean(), mean_enmu=x.enmu_base.mean(),
                         median_ratio=(x[ours] / x.enmu_base).median(),
                         per_dest_spearman=per_dest_spearman(off, ours, 'enmu_base')))
    x = off[['ours_nec', 'enmu_nec']].dropna()
    rows.append(dict(scope=label, compare='level ours_nec vs enmu_nec', n=len(x), pearson=x.ours_nec.corr(x.enmu_nec),
                     spearman=x.ours_nec.corr(x.enmu_nec, method='spearman'), mean_ours=x.ours_nec.mean(),
                     mean_enmu=x.enmu_nec.mean(), median_ratio=(x.ours_nec / x.enmu_nec).median(),
                     per_dest_spearman=per_dest_spearman(off, 'ours_nec', 'enmu_nec')))
    off = off.assign(d_ours=off.ours_nec - off.ours_base, d_enmu=off.enmu_nec - off.enmu_base)
    x = off[['d_ours', 'd_enmu']].dropna()
    rows.append(dict(scope=label, compare='delta nec-base', n=len(x), pearson=x.d_ours.corr(x.d_enmu),
                     spearman=x.d_ours.corr(x.d_enmu, method='spearman'), mean_ours=x.d_ours.mean(),
                     mean_enmu=x.d_enmu.mean(), median_ratio=np.nan,
                     per_dest_spearman=per_dest_spearman(off, 'd_ours', 'd_enmu'),
                     faster_ours=(x.d_ours < -0.5).mean(), faster_enmu=(x.d_enmu < -0.5).mean(),
                     slower_ours=(x.d_ours > 0.5).mean(), slower_enmu=(x.d_enmu > 0.5).mean()))
    return rows


def main():
    d = load()
    d.to_csv('%s/compare_df.csv' % OUT, index=False)
    acp = acp_codes()
    in_acp = d.ap_orig.str[:7].isin(acp) & d.ap_dest.str[:7].isin(acp)
    in_df = d.ap_orig.str[:2].eq('53') & d.ap_dest.str[:2].eq('53')  # SEMOB covers DF only, no ANTT semiurban
    rows = summary(d[in_df], 'df') + summary(d[in_acp], 'acp') + summary(d, 'all')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(pd.DataFrame(rows).round(3).to_string())
    diag = d[d.ap_orig == d.ap_dest]
    print('diagonal: enmu base median %.2f; ours base median %.1f, missing %d' %
          (diag.enmu_base.median(), diag.ours_base.median(), diag.ours_base.isna().sum()))
    for s in ('current', 'base', 'nec'):
        print('unreached share (%s, acp, off-diagonal, pop-weighted mean over pairs): %.3f' %
              (s, d[in_acp & (d.ap_orig != d.ap_dest)]['unreached_' + s].mean()))


if __name__ == '__main__':
    main()
