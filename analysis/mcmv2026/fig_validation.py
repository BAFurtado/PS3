"""Validation figure for the model paper: each area's 2017-19 level against its observed band and the calibration's
tolerance, in two panels: (a) levels the model places in or near the band in most areas, (b) levels it places below
the band in most areas.
usage (repo root, env ps3_2, MPLBACKEND=Agg): python analysis/mcmv2026/fig_validation.py [screen dir]
Input: <screen dir>/screen25_cities.csv from analysis/mcmv2026/validation/screen25.py on output/mcmv2026/d09 (default
output/mcmv2026/_analysis/screen25_27), bands from analysis/calibration/data/level_targets.csv and, for
unemployment and real GDP growth, analysis/mcmv2026/validation/gate_bounds25.csv.
x: position inside the band, (value - low) / (high - low): 0 is the lower limit, 1 the upper limit of each area's
band. Dark grey: the band. Light grey: the calibration's tolerance around it, I <= 3 with the seed term set to zero,
3 x 0.1 x the band's midpoint (median over areas where the band differs by area). Dots: blue in band, purple outside
the band with I <= 3, hollow orange with I > 3 (named). Held-out series have no I: blue in band, hollow orange outside.
A row goes to panel b when most areas lie below the band. Rows ordered by the share of areas in band.
Writes output/mcmv2026/_analysis/figures/fig_validation.{png,pdf}."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

SCREEN = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026/_analysis/screen25_27'
OUT = 'output/mcmv2026/_analysis/figures'

BLUE, PURPLE, ORANGE = '#2a78d6', '#9b59b6', '#eb6834'
INK, INK2, MUTED, GRID = '#1a1a19', '#52514e', '#8a8985', '#e6e5e1'
BAND, TOL = '#d6d5cf', '#eeede9'
L, M, S = 8.5, 7.5, 7
SSP = '/usr/share/texlive/texmf-dist/fonts/opentype/adobe/sourcesanspro/'
for f in ('Regular', 'Semibold', 'Bold', 'RegularIt'):
    if os.path.exists(SSP + f'SourceSansPro-{f}.otf'):
        font_manager.fontManager.addfont(SSP + f'SourceSansPro-{f}.otf')
plt.rcParams.update({
    'font.family': ['Source Sans Pro', 'DejaVu Sans'], 'font.size': S, 'axes.edgecolor': MUTED, 'axes.linewidth': .6,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': S, 'ytick.labelsize': M,
    'xtick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 0, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.spines.left': False, 'pdf.fonttype': 3, 'axes.unicode_minus': True})

NAMES = {'ARACAJU': 'Aracaju', 'BELEM': 'Belém', 'BELO HORIZONTE': 'Belo Horizonte', 'BOA VISTA': 'Boa Vista',
         'BRASILIA': 'Brasília', 'CAMPO GRANDE': 'Campo Grande', 'CUIABA': 'Cuiabá', 'CURITIBA': 'Curitiba',
         'FLORIANOPOLIS': 'Florianópolis', 'FORTALEZA': 'Fortaleza', 'GOIANIA': 'Goiânia',
         'JOAO PESSOA': 'João Pessoa', 'MACAPA': 'Macapá', 'MACEIO': 'Maceió', 'MANAUS': 'Manaus', 'NATAL': 'Natal',
         'PALMAS': 'Palmas', 'PORTO ALEGRE': 'Porto Alegre', 'PORTO VELHO': 'Porto Velho', 'RECIFE': 'Recife',
         'RIO BRANCO': 'Rio Branco', 'RIO DE JANEIRO': 'Rio de Janeiro', 'SALVADOR': 'Salvador',
         'SAO LUIS': 'São Luís', 'SAO PAULO': 'São Paulo', 'TERESINA': 'Teresina', 'VITORIA': 'Vitória'}

C = pd.read_csv(os.path.join(SCREEN, 'screen25_cities.csv')).set_index('acp')
gb = pd.read_csv('analysis/mcmv2026/validation/gate_bounds25.csv').set_index('acp').reindex(C.index)
lt = pd.read_csv('analysis/calibration/data/level_targets.csv')
B = {r.moment: (r.low, r.high) for r in lt[lt.region == 'ALL'].itertuples()}
one = lambda v: pd.Series(v, C.index)


def lvl(lab, col, band_txt, lo=None, hi=None, held_out=False):
    lo = one(B[col][0]) if lo is None else lo
    hi = one(B[col][1]) if hi is None else hi
    x = (C[col] - lo) / (hi - lo)
    tol = None if held_out else float((3 * .1 * (lo + hi) / 2 / (hi - lo)).median())
    I = None if held_out else C['I_' + col]
    return dict(lab=lab, x=x, tol=tol, I=I, band=band_txt, held_out=held_out)


ROWS = [
    lvl('Vacancy rate', 'vacancy', '0.08–0.13'),
    lvl('Dwellings built per 1,000 residents', 'housing_production_per_1000', '2–4'),
    lvl('House price over household income', 'price_income', '3.7–6.3'),
    lvl('House price over median wage', 'price_wage', '81–119'),
    lvl('Housing stock over permanent income', 'housing_stock_permanent_income', '3.1–4.7'),
    lvl('Household consumption over GDP', 'consumption_gdp', '0.55–0.65'),
    lvl('Non-tradable over tradable inflation', 'nontradable_inflation', '0–2.35 % a year'),
    lvl('Unemployment rate', 'unemployment', 'by area', gb[['census_u', 'pnad_min']].min(axis=1), gb.pnad_max),
    lvl('GDP per resident over IBGE, 2010', 'gdp_pc_y1', '0.8–1.2', one(.8), one(1.2), True),
    lvl('GDP per resident over IBGE, 2017–2019', 'gdp_pc_1719', '0.8–1.2', one(.8), one(1.2), True),
    lvl('Real GDP growth, 2012–2019', 'real_gdp_12_19', 'IBGE ± 1.5 pp a year',
        gb.real_gdp_12_19 - 1.5, gb.real_gdp_12_19 + 1.5, True),
]
for r in ROWS:
    r['in'] = ((r['x'] >= 0) & (r['x'] <= 1)).mean()
    r['below'] = (r['x'] < 0).mean()
PANELS = [('a', 'In or near the band in most areas', [r for r in ROWS if r['below'] <= .5]),
          ('b', 'Below the band in most areas', [r for r in ROWS if r['below'] > .5])]
for _, _, rows in PANELS:
    rows.sort(key=lambda r: (r['held_out'], -r['in']))

LEFT, RIGHT = -1.5, 1.8  # areas beyond RIGHT sit at the edge as a triangle, position in brackets


def swarm(x, width=.045, step=.16):
    """Vertical offsets so that dots closer than `width` in x do not overlap (greedy, centre out)."""
    order = np.argsort(x.values)
    placed, off = [], np.zeros(len(x))
    levels = [0] + [s * k * step for k in range(1, 6) for s in (1, -1)]
    for i in order:
        for lv in levels:
            if all(abs(x.values[i] - x.values[j]) >= width or abs(lv - off[j]) >= step * .95 for j in placed):
                off[i] = lv
                break
        placed.append(i)
    return off


def draw_row(ax, r, y):
    x, I = r['x'], r['I']
    if r['tol'] is not None:
        a, b = max(-r['tol'], LEFT - .08), min(1 + r['tol'], RIGHT + .1)
        ax.add_patch(plt.Rectangle((a, y - .42), b - a, .84, color=TOL, lw=0, zorder=0))
    ax.add_patch(plt.Rectangle((0, y - .42), 1, .84, color=BAND, lw=0, zorder=0))
    xc = x.clip(LEFT, RIGHT)
    off = swarm(xc)
    named, n = [], dict(blue=0, purple=0, hollow=0)
    for (a, xx), o in zip(xc.items(), off):
        inside = 0 <= x[a] <= 1
        near = I is not None and I[a] <= 3
        out = x[a] > RIGHT
        mk, ms = ('>', 4.6) if out else ('o', 4.2)
        if inside or near:
            col = BLUE if inside else PURPLE
            n['blue' if inside else 'purple'] += 1
            ax.plot(xx, y + o, mk, ms=ms, mew=.9, mfc=col, mec='white', zorder=3)
        else:
            n['hollow'] += 1
            ax.plot(xx, y + o, mk, ms=ms, mew=.9, mfc='white', mec=ORANGE, zorder=3)
            named.append((xx, NAMES[a] + (f' ({x[a]:.1f})' if out else '')))
    named.sort()
    clusters = []
    for xx, nm in named:
        if clusters and xx - clusters[-1][-1][0] < .25:
            clusters[-1].append((xx, nm))
        else:
            clusters.append([(xx, nm)])
    for cl in clusters:
        ax.text(np.mean([c[0] for c in cl]), y + .5, ', '.join(c[1] for c in cl), ha='center', va='top',
                fontsize=5.8, color=INK2, zorder=4)
    ax.annotate(r['lab'], (0, y), xycoords=('axes fraction', 'data'), xytext=(-6, 1.5), textcoords='offset points',
                ha='right', va='bottom', fontsize=M, color=INK2)
    ax.annotate('band ' + r['band'], (0, y), xycoords=('axes fraction', 'data'), xytext=(-6, -1.5),
                textcoords='offset points', ha='right', va='top', fontsize=S - .5, color=MUTED)
    return n


heights = [len(p[2]) + .55 * any(r['held_out'] for r in p[2]) for p in PANELS]
fig, axs = plt.subplots(2, 1, figsize=(6.3, 1.0 + .5 * sum(heights)), height_ratios=heights,
                        gridspec_kw={'hspace': .32})
counts = []
for ax, (letter, title, rows) in zip(axs, PANELS):
    y, prev_held = 0., False
    for r in rows:
        if r['held_out'] and not prev_held:
            y += .35
            ax.text(LEFT - .08, y - .45, 'Not calibrated, against IBGE', fontsize=S, color=MUTED, style='italic',
                    va='bottom')
            y += .2
            prev_held = True
        n = draw_row(ax, r, y)
        counts.append(dict(panel=letter, level=r['lab'], **n))
        y += 1.15
    ax.set_ylim(y - .25, -.65)
    ax.set_yticks([])
    ax.set_xlim(LEFT - .08, RIGHT + .1)
    for xv in (-1, -.5, .5, 1.5):
        ax.axvline(xv, color=GRID, lw=.5, zorder=-1)
    ax.annotate(letter, (0, 1), xycoords=('figure fraction', 'axes fraction'), xytext=(2, 6),
                textcoords='offset points', fontsize=L, fontweight='bold', color=INK, va='bottom')
    ax.annotate(title, (0, 1), xycoords=('figure fraction', 'axes fraction'), xytext=(13, 6),
                textcoords='offset points', fontsize=L, fontweight='semibold', color=INK, va='bottom')
    ax.set_xticks([-1, 0, 1], ['one band width\nbelow the band', 'lower\nlimit', 'upper\nlimit'])
leg = [Line2D([], [], ls='', marker='s', ms=7, mfc=BAND, mec=BAND, label='observed band'),
       Line2D([], [], ls='', marker='s', ms=7, mfc=TOL, mec=TOL, label='tolerance'),
       Line2D([], [], ls='', marker='o', ms=4.6, mfc=BLUE, mec='white', label='in band'),
       Line2D([], [], ls='', marker='o', ms=4.6, mfc=PURPLE, mec='white', label='within tolerance'),
       Line2D([], [], ls='', marker='o', ms=4.6, mfc='white', mec=ORANGE, label='beyond (named)')]
axs[0].legend(handles=leg, loc='lower right', bbox_to_anchor=(1.0, 1.06), ncol=5, frameon=False, fontsize=S,
              handletextpad=.25, columnspacing=1.0)
os.makedirs(OUT, exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig_validation.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
print(pd.DataFrame(counts).to_string(index=False))
# dots whose colour disagrees with the drawn zone (tolerance drawn without the seed term)
for r in ROWS:
    if r['I'] is None:
        continue
    t = r['tol']
    zone = (r['x'] >= -t) & (r['x'] <= 1 + t)
    bad = C.index[(zone & (r['I'] > 3)) | (~zone & (r['I'] <= 3))]
    if len(bad):
        print('zone/colour mismatch', r['lab'], [(a, round(r['x'][a], 2), round(r['I'][a], 2)) for a in bad])
