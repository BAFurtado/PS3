"""
Summarize a `main.py sensitivity` output directory into one tidy CSV.

Works for any single- or multi-param OFAT sweep (dirs named
PARAM_NAME=value/seed/stats.csv, as produced by `sensitivity PARAM:min:max:step`,
including multiple PARAM specs passed to a single `sensitivity` call).

Usage:
    python analysis/validation/summarize_sensitivity.py output/<run_id> > sensitivity_raw.csv

Writes the per-seed rows to stdout, and an aggregated (mean/std per param
value) version to <run_id>/sensitivity_summary_aggregated.csv.
"""
import sys
import glob
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from analysis.output import OUTPUT_DATA_SPEC

COLS = OUTPUT_DATA_SPEC["stats"]["columns"]

TAIL_START, TAIL_END = '2035-01-01', '2039-12-31'

OUTCOME_COLS = ['gini_index', 'unemployment', 'house_vacancy', 'pct_zero_consumption',
                'affordability_median', 'rent_default', 'families_helped',
                'amount_subsidised', 'perc_policy_money_spent']


def main(run_dir):
    rows = []
    for param_dir in sorted(glob.glob(os.path.join(run_dir, '*=*'))):
        param_name, param_value = os.path.basename(param_dir).split('=', 1)
        for seed_dir in sorted(glob.glob(os.path.join(param_dir, '*'))):
            if not os.path.isdir(seed_dir):
                continue
            stats_f = os.path.join(seed_dir, 'stats.csv')
            done_f = os.path.join(seed_dir, 'DONE')
            if not (os.path.exists(stats_f) and os.path.exists(done_f)):
                print(f"  skipping incomplete: {seed_dir}", file=sys.stderr)
                continue
            df = pd.read_csv(stats_f, sep=';', header=None, names=COLS)
            df['month_dt'] = pd.to_datetime(df['month'])
            tail = df[(df['month_dt'] >= TAIL_START) & (df['month_dt'] <= TAIL_END)]

            row = {'param_name': param_name, 'param_value': param_value,
                   'seed': os.path.basename(seed_dir)}
            for col in OUTCOME_COLS:
                if col == 'families_helped':
                    row[col] = tail[col].sum()          # total upgrades over the window
                else:
                    row[col] = tail[col].mean()
            rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(sys.stdout, index=False)

    agg = (res.groupby(['param_name', 'param_value'])
              .agg(n_seeds=('seed', 'count'),
                   **{f'{c}_mean': (c, 'mean') for c in OUTCOME_COLS},
                   **{f'{c}_std': (c, 'std') for c in OUTCOME_COLS})
              .reset_index())
    agg.to_csv(os.path.join(run_dir, 'sensitivity_summary_aggregated.csv'), index=False)
    print(f"\nAggregated summary written to {run_dir}/sensitivity_summary_aggregated.csv", file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analysis/validation/summarize_sensitivity.py output/<run_id>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])