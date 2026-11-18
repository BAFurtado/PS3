"""Figures for the model paper from mcmv2026: the intervention (d25) against the reference (d09).
usage (repo root, env ps3_2): python analysis/mcmv2026/figures.py [root]
Reads <root>/_analysis/{paths,jobs_yearly,hetero_effects,price_composition,screen_all,screen_city}.csv. Writes <root>/_analysis/figures/fig_paths.{png,pdf}
fig_whom.{png,pdf} and fig_cities.{png,pdf}.

fig_paths. Panel a: the MCMV budget as a share of the area's GDP, from world/mcmv_funds.py (0.25 % in 2010-16, 0.04 %
in 2017-22, 0.09 % in 2023-25) and runner_mcmv2026.py (d09: 0.09 % from 2026; d25: 0.25 % in 2026-30, 0.09 % from
2031). Panels b-e: yearly effect, pooled over 26 cities x 10 seeds, % of the reference, 90 % bootstrap over cities,
one shared y axis. Panels f and g each have their own y axis (the exception to one axis per row): jobs are the
construction and other-sector contributions to total employment (% of reference employment, January firm
snapshots), unemployment is in percentage points.
fig_whom: effect in 2031-34 (January snapshots 2032-34) by group, dot and 90 % interval, hollow dot when the interval
contains zero. Panels a and b share one x axis (%), owner share (percentage points) has its own. Income quintiles
hold equal family counts by construction, so they carry no families value.
fig_cities: per city, mean d25 - d09 effect over 10 seeds in 2031-34 (screen.py), 90 % interval from the seed standard
error, hollow dot when it contains zero, cities ordered by the dwellings effect, vertical line at the pooled effect.
Dwellings, families and residents share one x axis (%), unemployment (percentage points) and real GDP (wider seed
spread) each have their own. Under each panel: Spearman correlation of the city effects in seeds 0-4 against 5-9.
Panel e of fig_paths is the mean price of dwellings held by families outside MCMV (price_composition.py), because
stats.csv house_price averages over every dwelling and falls when cheaper MCMV units enter the stock. January snapshots
(jobs, this price) are plotted at the turn of the year (year - 0.5), annual means at mid-year.

Page: JASSS text width 16 cm, so every figure is 6.3 in wide and prints at 100 %, in the class's typeface (Source Sans
Pro, from TeX Live). Type scale: 8.5 pt for panel letters and titles, 7.5 pt for values and row labels, 7 pt for ticks, units and notes.
Units sit under the title, not on a rotated axis label."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager
from matplotlib.lines import Line2D

root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
A = os.path.join(root, '_analysis')
OUT = os.path.join(A, 'figures')
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID, SHADE = '#1a1a19', '#52514e', '#8a8985', '#e6e5e1', '#f2f1ed'
L, M, S = 8.5, 7.5, 7
DOSE = (2025.5, 2030.5)
MINUS = '−'
SSP = '/usr/share/texlive/texmf-dist/fonts/opentype/adobe/sourcesanspro/'
for f in ('Regular', 'Semibold', 'Bold', 'RegularIt'):
    if os.path.exists(SSP + f'SourceSansPro-{f}.otf'):
        font_manager.fontManager.addfont(SSP + f'SourceSansPro-{f}.otf')
plt.rcParams.update({
    'font.family': ['Source Sans Pro', 'DejaVu Sans'], 'font.size': S, 'axes.edgecolor': MUTED, 'axes.linewidth': .6,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': S, 'ytick.labelsize': S,
    'xtick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 0, 'xtick.major.pad': 3,
    'ytick.major.pad': 3, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'lines.linewidth': 1.6, 'pdf.fonttype': 3, 'svg.fonttype': 'none', 'axes.unicode_minus': True})
rng = np.random.default_rng(0)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def head(ax, letter, title, unit=None):
    """Panel letter and title on one line, the unit beneath in a smaller grey."""
    up = 15 if unit else 5
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(0, up), textcoords='offset points',
                fontsize=L, fontweight='bold', color=INK, va='bottom', ha='left')
    ax.annotate(title, (0, 1), xycoords='axes fraction', xytext=(11, up), textcoords='offset points',
                fontsize=L, fontweight='semibold', color=INK, va='bottom', ha='left')
    if unit:
        ax.annotate(unit, (0, 1), xycoords='axes fraction', xytext=(11, 4), textcoords='offset points',
                    fontsize=S, color=MUTED, va='bottom', ha='left')


def signed(v, unit):
    return ('+' if v >= 0 else MINUS) + f'{abs(v):.2f}{unit}'


def jobs_paths():
    """Construction and other-sector employment, d25 - d09 as % of d09 total employment, by January."""
    J = pd.read_csv(os.path.join(A, 'jobs_yearly.csv')).fillna(0)
    J['total'] = J.construction + J.other
    ref = J[J.scenario == 'd09'].set_index(['city', 'seed', 'year'])
    a = J[J.scenario == 'd25'].set_index(['city', 'seed', 'year'])
    idx = a.index.intersection(ref.index)
    d = (a.loc[idx, ['construction', 'other']] - ref.loc[idx, ['construction', 'other']]).groupby(['city', 'year']).mean()
    b = ref.loc[idx].total.groupby(['city', 'year']).mean()
    cities = d.index.get_level_values(0).unique()

    def stat(cs):
        dd = pd.concat([d.loc[c] for c in cs]).groupby(level=0).sum()
        bb = pd.concat([b.loc[c] for c in cs]).groupby(level=0).sum()
        return dd.div(bb, axis=0) * 100
    pt = stat(cities)
    bs = pd.concat([stat(cities[rng.integers(0, len(cities), len(cities))]) for _ in range(500)])
    lo, hi = bs.groupby(level=0).quantile(.05), bs.groupby(level=0).quantile(.95)
    rows = [dict(scenario='d25', var='jobs_' + v, year=y - .5, effect=pt.loc[y, v], p05=lo.loc[y, v],
                 p95=hi.loc[y, v]) for v in ('construction', 'other') for y in pt.index]
    return pd.DataFrame(rows)


def price_path():
    """Mean price of dwellings held by families outside MCMV, January snapshots (price_composition.py)."""
    T = pd.read_csv(os.path.join(A, 'price_composition.csv'))
    T = T[T.measure.str.startswith('dwellings of families') & T.window.str.isdigit()]
    return pd.DataFrame(dict(scenario='d25', var='price_existing', year=T.window.astype(int) - .5, effect=T.effect,
                             p05=T.p05, p95=T.p95))


def effect_frame(ax):
    ax.axvspan(*DOSE, color=SHADE, zorder=0, lw=0)
    ax.axhline(0, color=MUTED, lw=.7, zorder=1)
    ax.grid(axis='y', color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.set_xlim(2024.6, 2034.6)
    ax.set_xticks([2026, 2030, 2034])
    ax.spines['left'].set_visible(False)


def path(ax, P, var, color=BLUE, unit='%', end_label=True):
    p = P[(P['var'] == var) & (P.year >= 2025)].sort_values('year')
    ax.fill_between(p.year, p.p05, p.p95, color=color, alpha=.15, lw=0, zorder=2)
    ax.plot(p.year, p.effect, color=color, zorder=3, solid_capstyle='round', solid_joinstyle='round')
    last = p.iloc[-1]
    ax.plot(last.year, last.effect, 'o', color=color, ms=3.6, mec='white', mew=.9, zorder=4)
    if end_label:
        up = last.effect >= 0
        ax.annotate(signed(last.effect, unit), (last.year, last.effect), xytext=(0, 6 if up else -6),
                    textcoords='offset points', ha='right', va='bottom' if up else 'top', fontsize=M,
                    fontweight='semibold', color=INK)
    return p


def budget_panel(ax):
    years = np.arange(2010, 2035)
    hist = np.where(years <= 2016, .25, np.where(years <= 2022, .04, .09))
    ref = np.where(years <= 2025, hist, .09)
    itv = np.where(years <= 2025, hist, np.where(years <= 2030, .25, .09))
    edges = np.append(years, 2035)
    ax.axvspan(2026, 2031, color=SHADE, zorder=0, lw=0)
    ax.stairs(itv, edges, color=BLUE, lw=1.6, zorder=3, baseline=None)
    ax.stairs(ref, edges, color=MUTED, lw=1.6, zorder=4, baseline=None, ls=(0, (2.5, 1.5)))
    ax.stairs(hist[years <= 2025], edges[edges <= 2026], color=INK2, lw=1.6, zorder=5, baseline=None)
    ax.set_xlim(2010, 2035)
    ax.set_ylim(0, .3)
    ax.set_yticks([.04, .09, .25], ['0.04', '0.09', '0.25'])
    ax.set_xticks([2010, 2017, 2023, 2026, 2031, 2035])
    ax.grid(axis='y', color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    kw = dict(fontsize=M, color=INK2, ha='center')
    ax.text(2016.5, .26, 'both scenarios, 2010-25', va='bottom', **kw)
    ax.text(2028.5, .26, 'intervention 0.25%', va='bottom', fontweight='semibold', **{**kw, 'color': INK})
    ax.text(2028.5, .078, 'reference 0.09%', va='top', **kw)
    ax.text(2033, .1, 'both 0.09%', va='bottom', **kw)


def fig_paths():
    P = pd.concat([pd.read_csv(os.path.join(A, 'paths.csv')), jobs_paths(), price_path()])
    P = P[P.scenario == 'd25']
    fig = plt.figure(figsize=(6.3, 4.4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=.55)

    mid = gs[0].subgridspec(1, 4, wspace=.14)
    row = [('dwellings', 'Dwellings'), ('families', 'Families'), ('pop', 'Residents'), ('price_existing', 'House price')]
    axs = [fig.add_subplot(mid[0])]
    axs += [fig.add_subplot(mid[i], sharey=axs[0]) for i in (1, 2, 3)]
    for ax, (v, t), letter in zip(axs, row, 'abcd'):
        effect_frame(ax)
        path(ax, P, v)
        head(ax, letter, t, '% of the reference' if ax is axs[0] else
             'dwellings outside MCMV' if v == 'price_existing' else ' ')
        if ax is not axs[0]:
            ax.tick_params(labelleft=False)
    axs[0].set_ylim(-1.3, 1.05)
    axs[0].set_yticks(np.arange(-1.2, .81, .4))

    low = gs[1].subgridspec(1, 2, wspace=.22)
    ax = fig.add_subplot(low[0])
    effect_frame(ax)
    path(ax, P, 'jobs_construction', BLUE, end_label=False)
    path(ax, P, 'jobs_other', ORANGE, end_label=False)
    ax.legend([Line2D([], [], color=BLUE), Line2D([], [], color=ORANGE)], ['construction', 'other sectors'],
              loc='upper left', fontsize=M, handlelength=1.4, handletextpad=.5, borderaxespad=.2,
              labelcolor=INK2)
    ax.set_ylim(-.09, .25)
    head(ax, 'e', 'Jobs by sector', '% of reference employment')

    ax = fig.add_subplot(low[1])
    effect_frame(ax)
    path(ax, P, 'unemployment', unit=' pp')
    ax.set_ylim(-.15, .25)
    head(ax, 'f', 'Unemployment rate', 'percentage points')
    save(fig, 'fig_paths')


def fig_budget():
    """The two scenarios: MCMV budget as a share of the area's GDP, 2010-2034."""
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    budget_panel(ax)
    ax.set_ylabel("% of the area's GDP", fontsize=S, color=MUTED)
    save(fig, 'fig_budget')


def fig_whom():
    H = pd.read_csv(os.path.join(A, 'hetero_effects.csv'))
    H = H[(H.window == 'after') & (H.scenario == 'd25')]
    blocks = [(None, [('all', 'all', 'Metropolitan area')]),
              ('Municipality', [('zone', 'core', 'Core'), ('zone', 'periphery', 'Other municipalities')]),
              ('Neighbourhood', [('nbhd', 'poor', 'Poor'), ('nbhd', 'middle', 'Middle-income'),
                                 ('nbhd', 'rich', 'Rich')]),
              ('Family income quintile', [('quint', 'q1', '1 (poorest)'), ('quint', 'q2', '2'),
                                          ('quint', 'q3', '3'), ('quint', 'q4', '4'), ('quint', 'q5', '5 (richest)')])]
    rows, y, headers, pos = [], [], [], 0.
    for name, items in blocks:
        if name:
            headers.append((name, -pos))
            pos += 1
        for r in items:
            rows.append(r)
            y.append(-pos)
            pos += 1
        pos += .45
    panels = [('families', 'Families', '%', 1), ('members_per_family', 'Persons per family', '%', 1),
              ('renter_share', 'Owner share of families', 'percentage points', -1)]
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 3.7), sharey=True, gridspec_kw={'wspace': .12})
    for ax, (v, t, unit, sign), letter in zip(axs, panels, 'abc'):
        h = H[H['var'] == v].set_index(['group', 'value'])
        for yy, (g, val, _) in zip(y, rows):
            if (g, val) not in h.index or (v == 'families' and g == 'quint'):
                continue
            r = h.loc[(g, val)]
            e = sign * r.effect
            lo, hi = sorted((sign * r.p05, sign * r.p95))
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [yy, yy], color=BLUE, lw=2.6, alpha=.28, solid_capstyle='round', zorder=2)
            ax.plot(e, yy, 'o', ms=5, zorder=3, mew=1.2, color=BLUE, mfc=BLUE if sig else 'white', mec=BLUE)
            if g == 'all':
                ax.annotate(signed(e, '%' if unit == '%' else ' pp'), (e, yy), xytext=(-4 if e >= 0 else 4, 6),
                            textcoords='offset points', ha='left' if e >= 0 else 'right', va='bottom',
                            fontsize=M, fontweight='semibold', color=INK)
        if v == 'families':
            q = [yy for yy, r in zip(y, rows) if r[0] == 'quint']
            ax.text(0, np.mean(q), 'equal by\nconstruction', ha='center', va='center', fontsize=S, color=MUTED,
                    style='italic', linespacing=1.15,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        ax.axvline(0, color=MUTED, lw=.7, zorder=1)
        ax.grid(axis='x', color=GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        head(ax, letter, t, unit)
        if unit == '%':
            ax.set_xlim(-1.15, 1.15)
            ax.set_xticks([-1, -.5, 0, .5, 1], ['−1', '−0.5', '0', '0.5', '1'])
        else:
            ax.set_xlim(-.75, .75)
            ax.set_xticks([-.5, 0, .5], ['−0.5', '0', '0.5'])
    axs[0].set_yticks(y, [r[2] for r in rows], fontsize=M, color=INK2)
    axs[0].set_ylim(y[-1] - .7, .95)
    axs[0].get_yticklabels()[0].set_fontweight('semibold')
    axs[0].get_yticklabels()[0].set_color(INK)
    for name, yy in headers:
        axs[0].annotate(name, (0, yy), xycoords=('axes fraction', 'data'), xytext=(-3, 0),
                        textcoords='offset points', ha='right', va='center', fontsize=M, fontweight='semibold',
                        color=INK)
    save(fig, 'fig_whom')


NAMES = {'ARACAJU': 'Aracaju', 'BELEM': 'Belém', 'BELO HORIZONTE': 'Belo Horizonte', 'BOA VISTA': 'Boa Vista',
         'BRASILIA': 'Brasília', 'CAMPO GRANDE': 'Campo Grande', 'CUIABA': 'Cuiabá', 'CURITIBA': 'Curitiba',
         'FLORIANOPOLIS': 'Florianópolis', 'FORTALEZA': 'Fortaleza', 'GOIANIA': 'Goiânia',
         'JOAO PESSOA': 'João Pessoa', 'MACAPA': 'Macapá', 'MACEIO': 'Maceió', 'MANAUS': 'Manaus', 'NATAL': 'Natal',
         'PALMAS': 'Palmas', 'PORTO ALEGRE': 'Porto Alegre', 'PORTO VELHO': 'Porto Velho', 'RECIFE': 'Recife',
         'RIO BRANCO': 'Rio Branco', 'RIO DE JANEIRO': 'Rio de Janeiro', 'SALVADOR': 'Salvador',
         'SAO LUIS': 'São Luís', 'SAO PAULO': 'São Paulo', 'TERESINA': 'Teresina', 'VITORIA': 'Vitória'}


def fig_cities():
    C = pd.read_csv(os.path.join(A, 'screen_city.csv'))
    C = C[C.window == 'after']
    SA = pd.read_csv(os.path.join(A, 'screen_all.csv'))
    SA = SA[SA.window == 'after'].set_index('var')
    panels = [('number_domiciles', 'Dwellings', '%'), ('families_count', 'Families', '%'),
              ('pop', 'Residents', '%'), ('unemployment', 'Unemployment', 'percentage points'),
              ('gdp_level', 'Real GDP', '%')]
    order = C[C['var'] == 'number_domiciles'].sort_values('effect').city.tolist()
    y = {c: i for i, c in enumerate(order)}
    fig, axs = plt.subplots(1, 5, figsize=(6.3, 6.0), sharey=True,
                            gridspec_kw={'wspace': .16})
    for ax, (v, t, unit), letter in zip(axs, panels, 'abcde'):
        d = C[C['var'] == v].set_index('city')
        for c in order[::2]:
            ax.axhspan(y[c] - .5, y[c] + .5, color='#f6f5f2', lw=0, zorder=0)
        pooled = SA.loc[v, 'effect']
        ax.axvline(pooled, color=BLUE, lw=.9, alpha=.45, zorder=1)
        for c in order:
            e, se = d.loc[c, 'effect'], d.loc[c, 'se']
            lo, hi = e - 1.645 * se, e + 1.645 * se
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [y[c], y[c]], color=BLUE, lw=2.2, alpha=.28, solid_capstyle='round', zorder=2)
            ax.plot(e, y[c], 'o', ms=4, mew=1, color=BLUE, mfc=BLUE if sig else 'white', mec=BLUE, zorder=3)
        ax.axvline(0, color=MUTED, lw=.7, zorder=1)
        ax.grid(axis='x', color=GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        head(ax, letter, t, unit)
        ax.set_xlabel(f'seed halves \u03c1 = {SA.loc[v, "split_half"]:.2f}'.replace('-', MINUS), fontsize=S,
                      color=MUTED, labelpad=4)
        if v in ('number_domiciles', 'families_count', 'pop'):
            ax.set_xlim(-.75, 1.55)
            ax.set_xticks([-.5, 0, .5, 1, 1.5], [MINUS + '0.5', '0', '0.5', '1', '1.5'])
        elif v == 'unemployment':
            ax.set_xlim(-1.1, 1.1)
            ax.set_xticks([-1, 0, 1], [MINUS + '1', '0', '1'])
        else:
            ax.set_xlim(-2.6, 2.6)
            ax.set_xticks([-2, 0, 2], [MINUS + '2', '0', '2'])
    axs[0].annotate('all cities', (SA.loc['number_domiciles', 'effect'], len(order) - .4), xytext=(3, 0),
                    textcoords='offset points', fontsize=S, color=BLUE, va='bottom', ha='left')
    axs[0].set_yticks(range(len(order)), [NAMES[c] for c in order], fontsize=M, color=INK2)
    axs[0].set_ylim(-.7, len(order) - .3)
    save(fig, 'fig_cities')


fig_budget()
fig_paths()
fig_whom()
fig_cities()
print('written to', OUT)
