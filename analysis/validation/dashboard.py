"""
Calibration validation dashboard.

Loads all stats.csv files from a simulation output directory, stacks them into
a single file for reuse, and serves an interactive Dash dashboard for exploring
simulation output against calibration targets.

Usage
-----
    python analysis/validation/dashboard.py path/to/output_dir
    python analysis/validation/dashboard.py path/to/output_dir --force-reload
"""
import sys
import os
import json
import argparse

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from analysis.output import OUTPUT_DATA_SPEC

# ── Calibration reference moments (add observed values here when available) ───

OBSERVED = {
    'gdp_growth':   {'mean': 0.10,        'std': 0.018},
    'unemployment': {'mean': 0.090,       'std': 0.021},
    'gini_index':   {'mean': 0.540,       'std': 0.018},
    'inflation':    {'mean': 0.065 / 12,  'std': 0.031 / np.sqrt(12)},
}

STATS_COLUMNS = OUTPUT_DATA_SPEC['stats']['columns']
NUMERIC_COLUMNS = [c for c in STATS_COLUMNS if c != 'month']

STACKED_FILENAME = 'stacked_stats.csv'
BURN_IN_END = '2012-01-01'

# ── Data loading ──────────────────────────────────────────────────────────────

def _read_conf(run_dir):
    """Extract param_set label and city from conf.json in a run directory."""
    conf_path = os.path.join(run_dir, 'conf.json')
    if not os.path.exists(conf_path):
        return 'unknown', 'unknown'
    with open(conf_path) as f:
        data = json.load(f)
    acps = data.get('PARAMS', {}).get('PROCESSING_ACPS', ['unknown'])
    city = acps[0] if acps else 'unknown'
    return os.path.basename(os.path.dirname(run_dir)), city


def build_stacked(root_dir):
    """Walk root_dir, read all stats.csv files, return a tagged DataFrame."""
    frames = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'stats.csv' not in filenames:
            continue
        if 'DONE' not in filenames:
            pass#continue  # skip incomplete runs
        param_set, city = _read_conf(dirpath)
        run_id = os.path.relpath(dirpath, root_dir)
        try:
            df = pd.read_csv(
                os.path.join(dirpath, 'stats.csv'),
                sep=';', header=None, names=STATS_COLUMNS
            )
        except Exception:
            continue
        df['run_id']    = run_id
        df['param_set'] = param_set
        df['city']      = city
        frames.append(df)

    if not frames:
        raise ValueError(f"No completed stats.csv files found in {root_dir}.")

    stacked = pd.concat(frames, ignore_index=True)
    stacked['month'] = pd.to_datetime(stacked['month'])
    return stacked


def load_data(root_dir, force_reload=False):
    """Load stacked data from cache or rebuild it."""
    cache_path = os.path.join(root_dir, STACKED_FILENAME)
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=['month'])
        return df

    print("Building stacked dataset...")
    df = build_stacked(root_dir)
    df.to_csv(cache_path, index=False)
    print(f"Saved to {cache_path} ({len(df):,} rows, {df['run_id'].nunique()} runs)")
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(df, cities, param_sets, variable):
    """Return mean and ±2std band per (month, city, param_set) group."""
    mask = pd.Series(True, index=df.index)
    if cities and 'All' not in cities:
        mask &= df['city'].isin(cities)
    if param_sets and 'All' not in param_sets:
        mask &= df['param_set'].isin(param_sets)

    filtered = df.loc[mask, ['month', 'city', 'param_set', 'run_id', variable]].copy()
    grouped = (
        filtered
        .groupby(['month', 'city', 'param_set'])[variable]
        .agg(['mean', 'std'])
        .reset_index()
    )
    grouped['std'] = grouped['std'].fillna(0)
    return grouped


# ── Chart ─────────────────────────────────────────────────────────────────────

def make_figure(grouped, variable, show_burn_in=True):
    fig = go.Figure()
    groups = grouped.groupby(['city', 'param_set'])

    for (city, param_set), grp in groups:
        grp = grp.sort_values('month')
        label = f"{city} — {param_set}"
        color_idx = hash(label) % 10

        # CI band
        fig.add_trace(go.Scatter(
            x=pd.concat([grp['month'], grp['month'].iloc[::-1]]),
            y=pd.concat([grp['mean'] + 2 * grp['std'],
                         (grp['mean'] - 2 * grp['std']).iloc[::-1]]),
            fill='toself',
            fillcolor=f'rgba({50 + color_idx * 20}, {100 + color_idx * 15}, 200, 0.15)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=grp['month'],
            y=grp['mean'],
            name=label,
            mode='lines',
            line=dict(width=2),
        ))

    # Reference band (calibration target)
    if variable in OBSERVED:
        obs = OBSERVED[variable]
        fig.add_hrect(
            y0=obs['mean'] - 2 * obs['std'],
            y1=obs['mean'] + 2 * obs['std'],
            fillcolor='rgba(255, 180, 0, 0.15)',
            line=dict(width=0),
            annotation_text='Observed ±2σ',
            annotation_position='top left',
        )
        fig.add_hline(
            y=obs['mean'],
            line=dict(color='rgba(200, 140, 0, 0.8)', width=1.5, dash='dash'),
        )

    # Burn-in shaded region
    if show_burn_in:
        burn_in_ts = pd.Timestamp(BURN_IN_END)
        fig.add_vrect(
            x0=grouped['month'].min(), x1=burn_in_ts,
            fillcolor='rgba(150, 150, 150, 0.12)',
            line=dict(width=0),
            annotation_text='Burn-in',
            annotation_position='top left',
            annotation_font=dict(color='grey', size=11),
        )
        fig.add_vline(
            x=burn_in_ts,
            line=dict(color='rgba(100, 100, 100, 0.5)', width=1.2, dash='dot'),
        )

    fig.update_layout(
        xaxis_title='Month',
        yaxis_title=variable,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=40, b=40, l=60, r=20),
        hovermode='x unified',
        template='plotly_white',
    )
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def make_summary(grouped, variable):
    rows = []
    for (city, param_set), grp in grouped.groupby(['city', 'param_set']):
        rows.append({
            'City':      city,
            'Param set': param_set,
            'Mean':      f"{grp['mean'].mean():.4f}",
            'Std':       f"{grp['std'].mean():.4f}",
            'Min':       f"{grp['mean'].min():.4f}",
            'Max':       f"{grp['mean'].max():.4f}",
            'Observed mean': OBSERVED.get(variable, {}).get('mean', '—'),
        })
    return rows


# ── App ───────────────────────────────────────────────────────────────────────

def build_app(df):
    cities     = sorted(df['city'].unique())
    param_sets = sorted(df['param_set'].unique())
    variables  = NUMERIC_COLUMNS

    app = Dash(__name__)
    app.title = 'PS3 Calibration Dashboard'

    _dd_style  = {'width': '100%'}
    _opt_all   = [{'label': 'All', 'value': 'All'}]

    app.layout = html.Div([
        html.H2('PS3 Calibration Validation',
                style={'fontFamily': 'sans-serif', 'marginBottom': '8px'}),

        html.Div([
            html.Div([
                html.Label('City', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='city-select',
                    options=_opt_all + [{'label': c, 'value': c} for c in cities],
                    value=['All'],
                    multi=True,
                    style=_dd_style,
                ),
            ], style={'flex': 1, 'marginRight': '16px'}),

            html.Div([
                html.Label('Parameter set', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='param-select',
                    options=_opt_all + [{'label': p, 'value': p} for p in param_sets],
                    value=['All'],
                    multi=True,
                    style=_dd_style,
                ),
            ], style={'flex': 1, 'marginRight': '16px'}),

            html.Div([
                html.Label('Variable', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='var-select',
                    options=[{'label': v, 'value': v} for v in variables],
                    value='gdp_growth',
                    clearable=False,
                    style=_dd_style,
                ),
            ], style={'flex': 1, 'marginRight': '16px'}),

            html.Div([
                html.Label(' ', style={'fontWeight': 'bold', 'display': 'block'}),
                dcc.Checklist(
                    id='burn-in-toggle',
                    options=[{'label': ' Show burn-in period', 'value': 'show'}],
                    value=['show'],
                    style={'marginTop': '6px'},
                ),
            ], style={'flexShrink': 0, 'alignSelf': 'flex-end', 'paddingBottom': '2px'}),
        ], style={'display': 'flex', 'marginBottom': '16px',
                  'fontFamily': 'sans-serif'}),

        dcc.Graph(id='ts-chart', style={'height': '480px'}),

        html.H4('Summary', style={'fontFamily': 'sans-serif', 'marginTop': '16px'}),
        dash_table.DataTable(
            id='summary-table',
            style_table={'overflowX': 'auto'},
            style_cell={'fontFamily': 'sans-serif', 'fontSize': '13px',
                        'padding': '6px 12px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
        ),
    ], style={'maxWidth': '1200px', 'margin': '32px auto', 'padding': '0 24px'})

    @app.callback(
        Output('ts-chart', 'figure'),
        Output('summary-table', 'data'),
        Output('summary-table', 'columns'),
        Input('city-select', 'value'),
        Input('param-select', 'value'),
        Input('var-select', 'value'),
        Input('burn-in-toggle', 'value'),
    )
    def update(cities, param_sets, variable, burn_in_toggle):
        grouped       = aggregate(df, cities, param_sets, variable)
        show_burn_in  = bool(burn_in_toggle)
        fig           = make_figure(grouped, variable, show_burn_in=show_burn_in)
        rows          = make_summary(grouped, variable)
        cols          = [{'name': k, 'id': k} for k in rows[0].keys()] if rows else []
        return fig, rows, cols

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PS3 calibration validation dashboard')
    parser.add_argument('root_dir', help='Root output directory containing simulation runs')
    parser.add_argument('--force-reload', action='store_true',
                        help='Rebuild stacked_stats.csv even if it already exists')
    parser.add_argument('--port', type=int, default=8050)
    args = parser.parse_args()

    df = load_data(args.root_dir, force_reload=args.force_reload)
    app = build_app(df)
    print(f"\nDashboard running at http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port)
