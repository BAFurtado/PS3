"""
Main figure of the emissions paper: does the carbon tax cut emissions through innovation or contraction, where,
why, and at what cost?

    a  Emissions under the tax, % of NO_POLICY, all capitals pooled: total and its LMDI split into intensity
       (innovation) and scale (contraction), January of each year
    b  Map of the capitals: circle area = baseline emissions, colour = share of the cut that comes from innovation
    c  Why: share of baseline emissions in the sectors that can innovate (Utilities, Transport) against that share
    d  Cost by instrument: emissions against GDP, % of NO_POLICY, one point per tax scenario

Panels a-c compare TAX with NO_POLICY run by run (same capital and seed); d compares every tax scenario.
Bands and bars are 95% intervals across seeds of the pooled (all capitals) change.

    python -m analysis.emissions_paper.fig_main [ROOT] [--out DIR] [--window 2025 2029] [--outline SHP]

ROOT defaults to output/emissions. Writes fig_main.pdf, fig_main.png and the numbers behind them
(fig_main_paths.csv, fig_main_cities.csv, fig_main_cost.csv) to --out (default figures_emissions_paper/).
--outline: a shapefile of Brazil's states (e.g. IBGE BR_UF_2022) drawn under the map; without it the map shows
the capitals only.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats as sps

from analysis.emissions_paper import style
from analysis.emissions_paper.data import REFERENCE, load
from analysis.emissions_paper.decomposition import innovation_share, lmdi

SCENARIO = 'TAX'
POLICY_START = 2014 + 11 / 12          # December 2014: STARTING_DAY + ECO_POLICY_DAYS (runner_emissions.policy_start)
BASELINE_YEAR = 2014                    # January snapshot before the tax, identical in every scenario
RESPONSIVE = ['Utilities', 'Transport']  # the sectors whose tax per real of revenue is high enough to innovate
DEFAULT_OUTLINE = 'input/shapes/BR_UF_2022.shp'
MAP_LABELS = ['MANAUS', 'BRASILIA', 'RECIFE', 'RIO DE JANEIRO', 'PORTO ALEGRE']  # spread out, so they do not collide

SCENARIO_LABELS = {
    'TAX': 'Tax',
    'TAX_SUBSIDIES': 'Tax + subsidies',
    'TAX_RECYCLING': 'Tax + recycling',
    'TAX_RECYCLING_SUBSIDIES': 'Tax + recycling + subsidies',
}
DISPLAY = {
    'ARACAJU': 'Aracaju', 'BELEM': 'Belém', 'BELO HORIZONTE': 'Belo Horizonte', 'BOA VISTA': 'Boa Vista',
    'BRASILIA': 'Brasília', 'CAMPO GRANDE': 'Campo Grande', 'CUIABA': 'Cuiabá', 'CURITIBA': 'Curitiba',
    'FLORIANOPOLIS': 'Florianópolis', 'FORTALEZA': 'Fortaleza', 'GOIANIA': 'Goiânia', 'JOAO PESSOA': 'João Pessoa',
    'MACAPA': 'Macapá', 'MACEIO': 'Maceió', 'MANAUS': 'Manaus', 'NATAL': 'Natal', 'PALMAS': 'Palmas',
    'PORTO ALEGRE': 'Porto Alegre', 'PORTO VELHO': 'Porto Velho', 'RECIFE': 'Recife', 'RIO BRANCO': 'Rio Branco',
    'RIO DE JANEIRO': 'Rio de Janeiro', 'SALVADOR': 'Salvador', 'SAO LUIS': 'São Luís', 'SAO PAULO': 'São Paulo',
    'TERESINA': 'Teresina', 'VITORIA': 'Vitória',
}


# ── Numbers ───────────────────────────────────────────────────────────────────

def seed_interval(values):
    """Mean and 95% t-interval across seeds."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return pd.Series({'mean': v.mean() if len(v) else np.nan, 'lo': np.nan, 'hi': np.nan, 'n': len(v)})
    h = sps.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return pd.Series({'mean': v.mean(), 'lo': v.mean() - h, 'hi': v.mean() + h, 'n': len(v)})


def effects(batch, scenario=SCENARIO):
    """Sector-level LMDI effects of `scenario` against NO_POLICY, on the city-seed pairs finished in both."""
    pairs = batch.pairs(scenario)
    if not pairs:
        raise SystemExit('No {} run with a matching {} run'.format(scenario, REFERENCE))
    sec = batch.sectors[[(c, s) in pairs for c, s in zip(batch.sectors.city, batch.sectors.seed)]]
    return lmdi(sec[sec.scenario == scenario], sec[sec.scenario == REFERENCE])


def paths(eff):
    """Pooled change by year, % of NO_POLICY emissions: total, intensity, scale (mean and interval over seeds)."""
    by_seed = eff.groupby(['seed', 'year'])[['E_T', 'E_N', 'intensity', 'scale']].sum()
    pct = pd.DataFrame({'total': 100 * (by_seed.E_T / by_seed.E_N - 1),
                        'intensity': 100 * by_seed.intensity / by_seed.E_N,
                        'scale': 100 * by_seed.scale / by_seed.E_N})
    rows = []
    for (year, part), v in pct.stack().groupby(level=['year', 2]):
        rows.append({'year': year, 'part': part, **seed_interval(v.values)})
    return pd.DataFrame(rows)


def cities(eff, batch, window):
    """Per capital, over the window: innovation share of the cut, total change, baseline emissions, and the share of
    baseline emissions in RESPONSIVE sectors."""
    w = eff[eff.year.between(*window)].groupby(['city', 'seed'])[['E_T', 'E_N', 'intensity', 'scale']].sum()
    w['share'] = innovation_share(w.intensity, w.scale)
    w['total'] = 100 * (w.E_T / w.E_N - 1)
    out = w.groupby('city')[['share', 'total']].mean()

    base = batch.sectors[(batch.sectors.scenario == REFERENCE) & (batch.sectors.year == BASELINE_YEAR)]
    by_city = base.groupby(['city', 'seed']).E.sum().groupby('city').mean()
    resp = base[base.sector.isin(RESPONSIVE)].groupby(['city', 'seed']).E.sum().groupby('city').mean()
    out['baseline'] = by_city
    out['responsive'] = 100 * resp / by_city
    out['share'] *= 100
    return out.join(batch.coords.set_index('city')).reset_index()


def cost(batch, window):
    """Pooled emissions and GDP over the window, % of NO_POLICY, for every tax scenario (mean and interval)."""
    st = batch.stats[batch.stats.month.dt.year.between(*window)]
    rows = []
    for s in sorted(x for x in st.scenario.unique() if x != REFERENCE):
        pairs = batch.pairs(s)
        keep = st[[(c, d) in pairs for c, d in zip(st.city, st.seed)]]
        agg = keep.groupby(['scenario', 'seed', 'city'])[['emissions', 'gdp_level']].mean().groupby(
            ['scenario', 'seed']).sum()
        t, r = agg.xs(s, level='scenario'), agg.xs(REFERENCE, level='scenario')
        for col in ('emissions', 'gdp_level'):
            rows.append({'scenario': s, 'variable': col, **seed_interval(100 * (t[col] / r[col] - 1))})
    return pd.DataFrame(rows)


# ── Panels ────────────────────────────────────────────────────────────────────

def panel_paths(ax, p):
    parts = [('total', style.INK, 'total'), ('intensity', style.BLUE, 'innovation'),
             ('scale', style.ORANGE, 'contraction')]
    ends = []
    for part, color, label in parts:
        d = p[p.part == part].sort_values('year')
        ax.fill_between(d.year, d.lo, d.hi, color=color, alpha=0.15, lw=0)
        ax.plot(d.year, d['mean'], color=color, lw=2 if part == 'total' else 1.6, solid_capstyle='round',
                zorder=3 if part == 'total' else 2)
        last = d.iloc[-1]
        ax.plot(last.year, last['mean'], 'o', ms=4, color=color, mec='white', mew=0.8, zorder=4)
        ends.append((last.year, last['mean'], '{} {}'.format(label, style.fmt_pct(last['mean'])), color, part))
    style.zero_line(ax)
    lo, hi = ax.get_ylim()
    for (x, y, text, color, part), ly in zip(ends, style.spread([e[1] for e in ends], 0.07 * (hi - lo))):
        ax.annotate(text, xy=(x, y), xytext=(x + 0.6, ly), textcoords='data', va='center', fontsize=8,
                    color=color, fontweight='bold' if part == 'total' else 'normal', annotation_clip=False)
    ax.axvline(POLICY_START, color=style.MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.annotate('tax starts', xy=(POLICY_START, 0), xycoords=('data', 'axes fraction'), xytext=(-3, 3),
                textcoords='offset points', ha='right', va='bottom', fontsize=7.5, color=style.MUTED)
    ax.set_xlim(p.year.min() - 0.3, p.year.max() + 0.3)
    ax.set_xticks(range(int(p.year.min()), int(p.year.max()) + 1, 5))
    style.panel_title(ax, 'a', 'Emissions under the tax', '% of the no-policy run, all capitals, January of each year')


def panel_map(ax, c, outline=None):
    if outline:
        import geopandas as gpd
        gpd.read_file(outline).to_crs(4326).plot(ax=ax, color=style.LAND, edgecolor='white', lw=0.5, zorder=0)
    area = 900 * c.baseline / c.baseline.max()
    order = c.sort_values('baseline', ascending=False)
    ax.scatter(order.lon, order.lat, s=area.loc[order.index], c=order.share, cmap=style.INNOVATION_CMAP,
               vmin=0, vmax=100, edgecolor='white', lw=0.7, zorder=3)
    for _, r in c[c.city.isin(MAP_LABELS)].iterrows():
        ax.annotate(DISPLAY.get(r.city, r.city), xy=(r.lon, r.lat), xytext=(np.sqrt(area[r.name]) / 2 + 3, 0),
                    textcoords='offset points', ha='left', va='center', fontsize=7, color=style.TEXT, zorder=4)
    ax.set_xlim(-74.5, -33.5)
    ax.set_ylim(-34.5, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    bar = ax.inset_axes([0.04, 0.07, 0.36, 0.03])
    bar.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap=style.INNOVATION_CMAP)
    bar.set_xticks([0, 255])
    bar.set_xticklabels(['contraction', 'innovation'], fontsize=7, color=style.TEXT)
    bar.set_yticks([])
    for s in bar.spines.values():
        s.set_visible(False)
    bar.grid(False)
    bar.set_title('share of the cut', fontsize=7, color=style.MUTED, pad=3)
    style.panel_title(ax, 'b', 'Where the cut comes from', 'circle area: baseline emissions')


def panel_mix(ax, c):
    d = c.dropna(subset=['share', 'responsive'])
    ax.scatter(d.responsive, d.share, s=34, color=style.BLUE, edgecolor='white', lw=0.7, zorder=3)
    if len(d) > 2:
        slope, icept = np.polyfit(d.responsive, d.share, 1)
        xs = np.array([d.responsive.min(), d.responsive.max()])
        ax.plot(xs, icept + slope * xs, color=style.MUTED, lw=1, ls=(0, (4, 2)), zorder=2)
        rho = sps.spearmanr(d.responsive, d.share).statistic
        ax.annotate('Spearman ρ = {:.2f}'.format(rho), xy=(1, 1), xycoords='axes fraction', xytext=(0, -2),
                    textcoords='offset points', ha='right', va='top', fontsize=7.5, color=style.MUTED)
    # Name the capitals at the four corners of the cloud only
    z = (d[['responsive', 'share']] - d[['responsive', 'share']].mean()) / d[['responsive', 'share']].std().replace(0, 1)
    corners = [(z.responsive + z.share).idxmax(), (z.responsive + z.share).idxmin(),
               (z.responsive - z.share).idxmax(), (z.responsive - z.share).idxmin()]
    right = d.responsive.min() + 0.7 * (d.responsive.max() - d.responsive.min())
    for _, r in d.loc[list(dict.fromkeys(corners))].iterrows():
        # Labels on the right third go to the left of the dot, so they stay inside the panel
        left = r.responsive > right
        ax.annotate(DISPLAY.get(r.city, r.city), xy=(r.responsive, r.share), xytext=(-5 if left else 5, 3),
                    textcoords='offset points', ha='right' if left else 'left', fontsize=7, color=style.TEXT)
    ax.set_ylim(-4, 104)
    ax.grid(True, axis='both')
    ax.set_xlabel('Utilities and Transport, % of baseline emissions', fontsize=8)
    style.panel_title(ax, 'c', 'Why: what the capital emits', 'innovation share of the cut, %')


def panel_cost(cell, fig, k):
    """Two dot strips sharing the scenario rows, as in fig_whom: emissions and GDP, % of NO_POLICY. A filled dot
    is a change whose interval excludes zero."""
    order = [s for s in SCENARIO_LABELS if s in set(k.scenario)]
    sub = cell.subgridspec(1, 2, wspace=0.12)
    axes = [fig.add_subplot(sub[0, 0]), None]
    axes[1] = fig.add_subplot(sub[0, 1], sharey=axes[0])
    rows = np.arange(len(order))[::-1]
    for ax, variable, title in zip(axes, ('emissions', 'gdp_level'), ('Emissions', 'GDP')):
        d = k[k.variable == variable].set_index('scenario').loc[order]
        for y, (s, r) in zip(rows, d.iterrows()):
            if y % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color=style.BAND, lw=0, zorder=0)
            ax.plot([r.lo, r.hi], [y, y], color=style.BLUE_LIGHT, lw=4, solid_capstyle='round', zorder=2)
            filled = not (r.lo <= 0 <= r.hi)
            ax.plot(r['mean'], y, 'o', ms=6.5, mew=1.5, zorder=3, color=style.BLUE,
                    mfc=style.BLUE if filled else 'white')
        style.zero_line(ax, 'x')
        ax.grid(True, axis='x')
        ax.grid(False, axis='y')
        ax.set_ylim(-0.5, len(order) - 0.5)
        ax.set_xlabel(title + ', %', fontsize=8)
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([SCENARIO_LABELS[s] for s in order], fontsize=7.5)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    style.panel_title(axes[0], 'd', 'Cost by instrument', '% of the no-policy run, all capitals')


# ── Figure ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', nargs='?', default='output/emissions')
    ap.add_argument('--out', default='figures_emissions_paper')
    ap.add_argument('--window', nargs=2, type=int, default=[2025, 2029], metavar=('FIRST', 'LAST'),
                    help='years averaged in panels b-d (January snapshots for b-c, every month for d)')
    ap.add_argument('--outline', default=DEFAULT_OUTLINE if os.path.exists(DEFAULT_OUTLINE) else None)
    args = ap.parse_args()

    style.apply()
    batch = load(args.root)
    eff = effects(batch)
    p, c, k = paths(eff), cities(eff, batch, args.window), cost(batch, args.window)

    fig = plt.figure(figsize=(7.4, 7.0))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.1, 1], hspace=0.62, wspace=0.62,
                  left=0.08, right=0.95, top=0.9, bottom=0.08)
    panel_paths(fig.add_subplot(gs[0, 0]), p)
    panel_map(fig.add_subplot(gs[0, 1]), c, args.outline)
    panel_mix(fig.add_subplot(gs[1, 0]), c)
    panel_cost(gs[1, 1], fig, k)

    os.makedirs(args.out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(args.out, 'fig_main.' + ext))
    p.to_csv(os.path.join(args.out, 'fig_main_paths.csv'), index=False)
    c.to_csv(os.path.join(args.out, 'fig_main_cities.csv'), index=False)
    k.to_csv(os.path.join(args.out, 'fig_main_cost.csv'), index=False)
    pairs = len(batch.pairs(SCENARIO))
    print('fig_main from {} ({} TAX/NO_POLICY pairs, window {}-{}) -> {}'.format(
        args.root, pairs, *args.window, args.out))


if __name__ == '__main__':
    main()
