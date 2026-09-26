"""Is the fiscal leg of QLI a level shift, or is it chaotic divergence? (defect #5)

`compare_qli_arms.py` differences one matched-seed pair. That is the exact causal effect
of QLI_TAX_WEIGHT *in that world*, but at n = 1 it cannot be read as a statement about
the model: PS3 is chaotic, and §Reproducibility in model_defects.md measures two runs
differing only in seed at ~2% of a tail-window aggregate. An arm difference smaller than
that is not evidence of anything.

So take the same indicators over several seeds and put two quantities side by side:

  fiscal effect : mean over seeds of |Z(w=1, s) - Z(w=0, s)|, matched seed by seed
  noise floor   : mean over seed pairs of |Z(w=0, s_i) - Z(w=0, s_j)|

An indicator whose fiscal effect sits below its noise floor is comparable across arms —
which for house_price and the QLI path is exactly what the calibration is meant to buy.
One that rises above it is a real consequence of the fiscal leg.

    python analysis/validation/qli_arm_summary.py output/run__*/

Run directories are grouped by (city, QLI_TAX_WEIGHT, seed) from their own conf.json,
so any mix of arms can be passed in; only cities present at both weights are reported.
"""
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.output import columns_for  # noqa: E402

WATCH = ['average_qli', 'house_price', 'house_rent', 'price_level', 'gdp_level',
         'gini_index', 'unemployment', 'affordability_median']
TAIL_MONTHS = 60


def _one(run_dir, name):
    hits = sorted(p for p in Path(run_dir).rglob(name) if 'avg' not in p.parts)
    if len(hits) != 1:
        raise SystemExit('{}: expected one {}, found {}'.format(run_dir, name, len(hits)))
    return hits[0]


def _tail_means(run_dir):
    params = json.loads(_one(run_dir, 'conf.json').read_text())['PARAMS']
    stats = pd.read_csv(_one(run_dir, 'stats.csv'), sep=';', header=None)
    stats.columns = columns_for('stats', stats.shape[1])
    stats['month'] = pd.to_datetime(stats['month'])
    tail = stats.sort_values('month').tail(TAIL_MONTHS)
    key = (','.join(params['PROCESSING_ACPS']),
           float(params.get('QLI_TAX_WEIGHT', 0.0)),
           params['SEED'])
    return key, {c: tail[c].mean() for c in WATCH if c in tail.columns}


def main(run_dirs):
    runs = dict(_tail_means(d) for d in run_dirs)

    cities = sorted({k[0] for k in runs})
    for city in cities:
        weights = sorted({k[1] for k in runs if k[0] == city})
        if len(weights) < 2:
            print('\n{}: only w = {} present, skipping'.format(city, weights))
            continue
        w_lo, w_hi = weights[0], weights[-1]

        matched = sorted({k[2] for k in runs if (city, w_lo, k[2]) in runs
                          and (city, w_hi, k[2]) in runs})
        base_seeds = sorted({k[2] for k in runs if k[:2] == (city, w_lo)})

        print('\n{}  —  w = {} vs w = {}'.format(city, w_lo, w_hi))
        print('  matched seeds: {}   baseline seeds for the floor: {}'
              .format(len(matched), len(base_seeds)))
        if not matched:
            print('  no seed present in both arms; nothing is identified')
            continue
        if len(base_seeds) < 2:
            print('  only one baseline seed — the noise floor cannot be measured, so '
                  'the effects below have nothing to be compared against')

        rows = []
        for col in WATCH:
            effects = [abs(runs[(city, w_hi, s)][col] - runs[(city, w_lo, s)][col])
                       / abs(runs[(city, w_lo, s)][col]) * 100
                       for s in matched if runs[(city, w_lo, s)].get(col)]
            floor = [abs(runs[(city, w_lo, a)][col] - runs[(city, w_lo, b)][col])
                     / abs(runs[(city, w_lo, a)][col]) * 100
                     for a, b in itertools.combinations(base_seeds, 2)
                     if runs[(city, w_lo, a)].get(col)]
            if not effects:
                continue
            eff = sum(effects) / len(effects)
            flr = sum(floor) / len(floor) if floor else float('nan')
            rows.append({
                'indicator': col,
                'w{}'.format(w_lo): sum(runs[(city, w_lo, s)][col]
                                        for s in base_seeds) / len(base_seeds),
                'w{}'.format(w_hi): sum(runs[(city, w_hi, s)][col]
                                        for s in matched) / len(matched),
                'fiscal_effect_%': eff,
                'seed_noise_%': flr,
                'verdict': ('—' if flr != flr
                            else 'comparable' if eff <= flr else 'ABOVE NOISE'),
            })
        print(pd.DataFrame(rows).set_index('indicator').round(4).to_string())

    print('\nlast {} months. fiscal_effect is |w_hi - w_lo| at matched seed, averaged '
          'over seeds;'.format(TAIL_MONTHS))
    print('seed_noise is |w_lo - w_lo| across seed pairs. "comparable" means the arm '
          'difference')
    print('is not distinguishable from chaotic divergence at this seed count.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])