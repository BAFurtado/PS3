"""A/B two run directories that differ only in QLI_TAX_WEIGHT (defect #5, step 4).

The fiscal leg of the QLI driver is only usable as a designed arm if turning it on at
the calibrated QLI_SPEND_NORM leaves the baseline essentially where it was — otherwise
w is not a policy lever but a different model, and nothing run at w = 0 stays comparable.
The acceptance test on record is: QLI path and house_price within ~1%.

    python analysis/validation/compare_qli_arms.py <w0_dir> <w1_dir>

Both arms must share a seed; the script says so if they do not. Reported over the whole
horizon and over the last 60 months, because a driver difference compounds and a
tail-window check is the one that binds.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.output import columns_for  # noqa: E402

STATS_WATCH = ['house_price', 'average_qli', 'price_level', 'gdp_level',
               'gini_index', 'unemployment', 'house_rent', 'affordability_median']
TAIL_MONTHS = 60
TOLERANCE = 0.01  # the ~1% on record


def _one(run_dir, name):
    """The single run's own output file, never the `avg/` copy of it."""
    hits = sorted(p for p in Path(run_dir).rglob(name) if 'avg' not in p.parts)
    if not hits:
        raise SystemExit('no {} under {}'.format(name, run_dir))
    if len(hits) > 1:
        raise SystemExit('{} has {} runs; this compares one run per arm'
                         .format(run_dir, len(hits)))
    return hits[0]


def _read(run_dir):
    run_dir = Path(run_dir)
    conf = json.loads(_one(run_dir, 'conf.json').read_text())
    params = conf['PARAMS']

    stats = pd.read_csv(_one(run_dir, 'stats.csv'), sep=';', header=None)
    stats.columns = columns_for('stats', stats.shape[1])
    stats['month'] = pd.to_datetime(stats['month'])

    regional = pd.read_csv(_one(run_dir, 'regional.csv'), sep=';', header=None)
    regional.columns = columns_for('regional', regional.shape[1])
    regional['month'] = pd.to_datetime(regional['month'])
    return params, stats.set_index('month'), regional


def _pct(a, b):
    return float('nan') if a == 0 else (b - a) / abs(a) * 100


def main(w0_dir, w1_dir):
    p0, s0, r0 = _read(w0_dir)
    p1, s1, r1 = _read(w1_dir)

    print('arm 0: {:<10} w = {}  seed = {}'.format(
        ','.join(p0['PROCESSING_ACPS']), p0.get('QLI_TAX_WEIGHT'), p0['SEED']))
    print('arm 1: {:<10} w = {}  seed = {}  QLI_SPEND_NORM = {}'.format(
        ','.join(p1['PROCESSING_ACPS']), p1.get('QLI_TAX_WEIGHT'), p1['SEED'],
        p1.get('QLI_SPEND_NORM')))
    if p0['SEED'] != p1['SEED']:
        print('\n!! seeds differ — this is not a matched-seed A/B and the differences '
              'below mix the fiscal leg with chaotic divergence')
    if p0['PROCESSING_ACPS'] != p1['PROCESSING_ACPS']:
        raise SystemExit('different cities; nothing to compare')

    common = s0.index.intersection(s1.index)
    tail = common[-TAIL_MONTHS:]

    rows = []
    for col in STATS_WATCH:
        if col not in s0.columns:
            continue
        rows.append({
            'indicator': col,
            'w0_full': s0.loc[common, col].mean(),
            'w1_full': s1.loc[common, col].mean(),
            'delta_%_full': _pct(s0.loc[common, col].mean(), s1.loc[common, col].mean()),
            'w0_tail': s0.loc[tail, col].mean(),
            'w1_tail': s1.loc[tail, col].mean(),
            'delta_%_tail': _pct(s0.loc[tail, col].mean(), s1.loc[tail, col].mean()),
        })
    table = pd.DataFrame(rows).set_index('indicator')
    print('\nmeans over the full horizon and over the last {} months\n'.format(TAIL_MONTHS))
    print(table.round(4).to_string())

    # The acceptance test is on the two the recipe names.
    print('\nacceptance (|delta| <= {:.0%} on the tail window):'.format(TOLERANCE))
    verdict = True
    for col in ('average_qli', 'house_price'):
        if col not in table.index:
            continue
        d = abs(table.loc[col, 'delta_%_tail']) / 100
        ok = d <= TOLERANCE
        verdict &= ok
        print('  {:<14} {:+.3f}%   {}'.format(col, table.loc[col, 'delta_%_tail'],
                                              'PASS' if ok else 'FAIL'))
    print('  => {}'.format('comparable' if verdict
                           else 'NOT comparable at this QLI_SPEND_NORM'))

    # Where the fiscal leg is supposed to show up: municipalities whose spending per
    # capita is high relative to their value added should gain QLI relative to w = 0.
    # If this is flat the leg is wired but inert.
    key = ['month', 'mun_id']
    m = (r0[key + ['qli_index', 'qli_gdp_pc', 'qli_spend_pc']]
         .merge(r1[key + ['qli_index']], on=key, suffixes=('_w0', '_w1')))
    last = m[m['month'] == m['month'].max()].copy()
    if len(last) > 1:
        hist = m[m['qli_gdp_pc'] > 0].copy()
        hist['ratio'] = hist['qli_spend_pc'] / hist['qli_gdp_pc']
        ratio_by_mun = hist.groupby('mun_id')['ratio'].mean()
        last = last.set_index('mun_id')
        last['spend_gdp_ratio'] = ratio_by_mun
        last['qli_delta_%'] = (last['qli_index_w1'] - last['qli_index_w0']) \
            / last['qli_index_w0'] * 100
        last = last.sort_values('spend_gdp_ratio')
        print('\nper-municipality at the final month, sorted by spend/GDP ratio')
        print(last[['spend_gdp_ratio', 'qli_index_w0', 'qli_index_w1',
                    'qli_delta_%']].round(4).to_string())
        if last['spend_gdp_ratio'].std() > 0:
            r = last['spend_gdp_ratio'].corr(last['qli_delta_%'])
            print('\ncorr(spend/GDP ratio, QLI gain from the fiscal leg) = {:.3f}'.format(r))
            print('positive is the mechanism: fiscally favoured municipalities gain QLI, '
                  'and through region.index, house prices')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])