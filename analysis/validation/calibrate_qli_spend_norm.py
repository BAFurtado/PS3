"""Calibrate QLI_SPEND_NORM, the fiscal counterpart of QLI_GDP_NORM (defect #5).

The QLI driver is

    driver = (1 - w) * sqrt(gdp_pc / QLI_GDP_NORM) + w * sqrt(spend_pc / QLI_SPEND_NORM)

with w = QLI_TAX_WEIGHT. Setting

    QLI_SPEND_NORM = mean(spend_pc / gdp_pc) * QLI_GDP_NORM

makes the two legs numerically equal for a municipality spending the average share of
its value added, so moving w does not shift the baseline. The fiscal leg is then valuable
not because it changes levels but because it creates a LEVER: FPM is redistributive, so
spend per capita moves across municipalities independently of GDP per capita.

Run this against w = 0 output directories — the flow accumulator records spending whether
or not the leg is weighted in, so the calibration sample is the untreated model.

    python analysis/validation/calibrate_qli_spend_norm.py output/run__*/

Both columns come straight out of Region.update_qli, so this measures the ratio the
formula actually sees, not a reconstruction of it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.output import columns_for  # noqa: E402

# Ignore the first year: QLI is initialised from IDHM and municipal GDP takes a few
# months to settle, so the opening ratios are transient rather than structural.
WARMUP_MONTHS = 12


def _load(run_dir):
    """Every regional.csv under a run directory, with its city and its params."""
    run_dir = Path(run_dir)
    confs = list(run_dir.rglob('conf.json'))
    if not confs:
        raise SystemExit('no conf.json under {}'.format(run_dir))
    conf = json.loads(confs[0].read_text())
    city = ','.join(conf['PARAMS']['PROCESSING_ACPS'])
    w = conf['PARAMS'].get('QLI_TAX_WEIGHT', 0.0)

    # `avg/` holds the mean across runs and repeats every municipality-month already
    # present in the numbered run directories; including it double-counts, and with
    # n > 1 it would silently over-weight the mean it is itself derived from.
    frames = []
    for path in sorted(p for p in run_dir.rglob('regional.csv')
                       if 'avg' not in p.parts):
        df = pd.read_csv(path, sep=';', header=None, decimal='.')
        df.columns = columns_for('regional', df.shape[1])
        if 'qli_spend_pc' not in df.columns:
            raise SystemExit(
                '{} predates the QLI drivers ({} fields). Re-run with the fiscal leg '
                'in place — the ratio cannot be recovered from an older layout.'
                .format(path, df.shape[1]))
        df['run'] = path.parent.name
        frames.append(df)
    if not frames:
        raise SystemExit('no regional.csv under {}'.format(run_dir))
    out = pd.concat(frames, ignore_index=True)
    out['city'] = city
    out['w'] = w
    return out


def main(run_dirs):
    df = pd.concat([_load(d) for d in run_dirs], ignore_index=True)
    df['month'] = pd.to_datetime(df['month'])
    start = df['month'].min() + pd.DateOffset(months=WARMUP_MONTHS)
    df = df[df['month'] >= start]

    # A municipality-month with no value added has no meaningful ratio; drop rather
    # than let a division by ~0 dominate the mean.
    df = df[df['qli_gdp_pc'] > 0].copy()
    df['ratio'] = df['qli_spend_pc'] / df['qli_gdp_pc']

    qli_gdp_norm = 3.5  # conf/default/params.py

    print('QLI_SPEND_NORM calibration — spend_pc / gdp_pc, municipality-months')
    print('excluding the first {} months; {} municipality-months\n'
          .format(WARMUP_MONTHS, len(df)))

    per_city = df.groupby('city')['ratio'].agg(['mean', 'median', 'std', 'count'])
    per_city['implied_norm'] = per_city['mean'] * qli_gdp_norm
    print(per_city.round(4).to_string())

    # Weight cities equally rather than by municipality count, so Goiânia's 15
    # municipalities do not set the norm on their own.
    pooled = per_city['mean'].mean()
    print('\nmean of city means      : {:.4f}'.format(pooled))
    print('pooled mean over months : {:.4f}'.format(df['ratio'].mean()))
    print('pooled median           : {:.4f}'.format(df['ratio'].median()))
    print('\nQLI_SPEND_NORM = {:.4f} x {} = {:.3f}'
          .format(pooled, qli_gdp_norm, pooled * qli_gdp_norm))

    # How much of the driver's cross-sectional variation is fiscal rather than
    # economic: if spend_pc were a fixed multiple of gdp_pc the leg would be a
    # relabelling, and the correlation below would be 1.
    by_mun = df.groupby(['city', 'mun_id'])[['qli_gdp_pc', 'qli_spend_pc']].mean()
    if len(by_mun) > 2:
        r = np.corrcoef(by_mun['qli_gdp_pc'], by_mun['qli_spend_pc'])[0, 1]
        print('\ncross-municipality corr(gdp_pc, spend_pc) = {:.3f} over {} municipalities'
              .format(r, len(by_mun)))
        print('(1.0 would mean the fiscal leg is a relabelling of the economic one)')
        print('ratio spread across municipalities: sd = {:.4f}, min = {:.4f}, max = {:.4f}'
              .format(by_mun.eval('qli_spend_pc / qli_gdp_pc').std(),
                      by_mun.eval('qli_spend_pc / qli_gdp_pc').min(),
                      by_mun.eval('qli_spend_pc / qli_gdp_pc').max()))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])