"""
OAT (One-At-a-Time) sensitivity summary for PolicySpace3.

Scans output/ for sensitivity runs — subfolders whose name contains '='
(produced by `python main.py sensitivity PARAM:min:max:steps`) — and
generates:

  sensitivity_summary.csv            last-period medians per (config, run)
  analysis/sensitivity_plots/
    heatmap.png                      normalised impact matrix (params × indicators)
    sweep_<PARAM>.png                indicator vs param value, one per swept parameter

Usage:
    python analysis/sensitivity_oat.py [OUTPUT_DIR]

OUTPUT_DIR defaults to <project_root>/output.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from analysis.output import OUTPUT_DATA_SPEC

# ── configuration ─────────────────────────────────────────────────────────────

KEY_INDICATORS = [
    ('house_vacancy',      'Vacancy rate'),
    ('house_price',        'House price'),
    ('house_rent',         'Rent'),
    ('unemployment',       'Unemployment'),
    ('gini_index',         'Gini'),
    ('gdp_level',          'GDP level'),
    ('affordable',         'Affordability'),
    ('inflation',          'Inflation'),
    ('loan_approval_rate', 'Loan approval'),
    ('rent_default',       'Rent default'),
    ('average_qli',        'Avg QLI'),
]

LAST_N_MONTHS = 24
STATS_COLS = OUTPUT_DATA_SPEC['stats']['columns']

# ── data loading ──────────────────────────────────────────────────────────────

def _config_part(path: Path) -> str | None:
    """Return the folder name that encodes the parameter configuration."""
    for part in reversed(path.parts):
        if '=' in part and not part.endswith('.csv'):
            return part
    return None


def _parse_config(config_str: str) -> dict:
    """'P=v' or 'P1=v1__P2=v2' → {P: v, ...}."""
    meta = {}
    for segment in config_str.split('__'):
        if '=' in segment:
            k, v = segment.split('=', 1)
            meta[k] = v
    return meta


def load_sensitivity_data(output_dir: Path) -> pd.DataFrame:
    files = [
        p for p in output_dir.rglob('stats.csv')
        if 'avg' not in p.parts
        and any('=' in part and not part.endswith('.csv') for part in p.parts)
    ]

    if not files:
        raise FileNotFoundError(
            f'No sensitivity stats.csv files found under {output_dir}.\n'
            f'Run `python main.py sensitivity PARAM:min:max:steps` first.'
        )

    dfs = []
    for p in files:
        config_str = _config_part(p)
        if config_str is None:
            continue
        meta = _parse_config(config_str)
        try:
            df = pd.read_csv(
                p, sep=';', header=None, names=STATS_COLS,
                na_values=['inf', '-inf', 'Inf', '-Inf', 'nan'],
            )
        except Exception as exc:
            warnings.warn(f'Skipping {p}: {exc}')
            continue

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['month'] = pd.to_datetime(df['month'], errors='coerce')

        for k, v in meta.items():
            df[k] = v
        df['run_id'] = p.parent.name
        df['config_str'] = config_str
        dfs.append(df)

    if not dfs:
        raise RuntimeError('All sensitivity files were unreadable.')

    return pd.concat(dfs, ignore_index=True)


# ── aggregation ───────────────────────────────────────────────────────────────

def last_period_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Median of the last LAST_N_MONTHS rows per (config_str, run_id)."""
    ind_cols = [k for k, _ in KEY_INDICATORS]
    param_cols = [
        c for c in data.columns
        if c not in STATS_COLS and c not in ('config_str', 'run_id')
    ]

    rows = []
    for (config_str, run_id), grp in data.groupby(['config_str', 'run_id']):
        tail = grp.tail(LAST_N_MONTHS)
        row: dict = {'config_str': config_str, 'run_id': run_id}
        for pc in param_cols:
            row[pc] = grp[pc].iloc[0]
        for ind in ind_cols:
            if ind in tail.columns:
                row[ind] = tail[ind].median()
        rows.append(row)

    return pd.DataFrame(rows)


def detect_oat_params(summary: pd.DataFrame) -> list[str]:
    ind_cols = {k for k, _ in KEY_INDICATORS}
    skip = ind_cols | {'config_str', 'run_id'}
    return [
        c for c in summary.columns
        if c not in skip and summary[c].nunique(dropna=True) > 1
    ]


# ── plotting ──────────────────────────────────────────────────────────────────

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def plot_sweep(param: str, summary: pd.DataFrame,
               avail_indicators: list[tuple], plots_dir: Path):
    """Key indicators vs param value: median line + IQR band."""
    sub = summary[summary[param].notna()].copy()
    sub['_pval'] = _to_numeric(sub[param])
    sub = sub.dropna(subset=['_pval'])
    if sub.empty or sub['_pval'].nunique() < 2:
        return

    ncols = 3
    nrows = (len(avail_indicators) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, (ind, label) in zip(axes_flat, avail_indicators):
        grouped = sub.groupby('_pval')[ind]
        agg = pd.DataFrame({
            'med': grouped.median(),
            'q1':  grouped.quantile(0.25),
            'q3':  grouped.quantile(0.75),
        }).reset_index().sort_values('_pval')

        ax.plot(agg['_pval'], agg['med'], marker='o', color='steelblue', lw=1.5)
        ax.fill_between(agg['_pval'], agg['q1'], agg['q3'],
                        alpha=0.25, color='steelblue')
        ax.set_title(label, fontsize=9, pad=4)
        ax.set_xlabel(param, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.4, alpha=0.5)

    for ax in axes_flat[len(avail_indicators):]:
        ax.set_visible(False)

    fig.suptitle(f'OAT sweep — {param}', fontsize=12, y=1.02)
    fig.tight_layout()
    out = plots_dir / f'sweep_{param}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  sweep saved → {out.name}')


def plot_heatmap(summary: pd.DataFrame, oat_params: list[str],
                 avail_indicators: list[tuple], plots_dir: Path):
    """
    Normalised impact matrix: rows = parameters, cols = indicators.
    Cell value = (max_median - min_median) / overall_std for that indicator.
    """
    ind_keys  = [k for k, _ in avail_indicators]
    ind_labels = [lb for _, lb in avail_indicators]

    matrix = np.zeros((len(oat_params), len(ind_keys)))

    for i, param in enumerate(oat_params):
        sub = summary[summary[param].notna()].copy()
        sub['_pval'] = _to_numeric(sub[param])
        sub = sub.dropna(subset=['_pval'])
        for j, ind in enumerate(ind_keys):
            agg = sub.groupby('_pval')[ind].median()
            if len(agg) < 2 or agg.isna().all():
                continue
            overall_std = summary[ind].std(ddof=1)
            if overall_std and not np.isnan(overall_std) and overall_std > 0:
                matrix[i, j] = (agg.max() - agg.min()) / overall_std
            else:
                matrix[i, j] = agg.max() - agg.min()

    fig, ax = plt.subplots(figsize=(max(8, len(ind_keys) * 0.9),
                                    max(4, len(oat_params) * 0.5)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label='(max − min) / std')

    ax.set_xticks(range(len(ind_keys)))
    ax.set_xticklabels(ind_labels, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(oat_params)))
    ax.set_yticklabels(oat_params, fontsize=8)
    ax.set_title('OAT sensitivity — normalised impact matrix', fontsize=11)

    # annotate cells
    for i in range(len(oat_params)):
        for j in range(len(ind_keys)):
            v = matrix[i, j]
            if v > 0:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6.5,
                        color='white' if v > matrix.max() * 0.7 else 'black')

    fig.tight_layout()
    out = plots_dir / 'heatmap.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  heatmap saved → {out.name}')


# ── entry point ───────────────────────────────────────────────────────────────

def main(output_dir: Path | None = None):
    output_dir = output_dir or (ROOT / 'output')
    plots_dir  = ROOT / 'analysis' / 'sensitivity_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f'Scanning {output_dir} for sensitivity runs …')
    data = load_sensitivity_data(output_dir)
    n_configs = data['config_str'].nunique()
    n_files   = data.groupby(['config_str', 'run_id']).ngroups
    print(f'  {n_files} run-files across {n_configs} parameter configurations')

    print(f'Aggregating last {LAST_N_MONTHS} months per run …')
    summary = last_period_summary(data)
    csv_out = ROOT / 'sensitivity_summary.csv'
    summary.to_csv(csv_out, index=False)
    print(f'  saved → {csv_out.name}  ({len(summary)} rows)')

    oat_params = detect_oat_params(summary)
    if not oat_params:
        print('No OAT parameters with multiple values detected — nothing to plot.')
        return

    print(f'  parameters: {oat_params}')

    avail = [(k, lb) for k, lb in KEY_INDICATORS if k in summary.columns
             and summary[k].notna().any()]

    print(f'Generating sweep plots ({len(oat_params)} parameters) …')
    for param in oat_params:
        plot_sweep(param, summary, avail, plots_dir)

    print('Generating impact heatmap …')
    plot_heatmap(summary, oat_params, avail, plots_dir)

    print(f'\nDone.  All output in {plots_dir}')


if __name__ == '__main__':
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(arg)
