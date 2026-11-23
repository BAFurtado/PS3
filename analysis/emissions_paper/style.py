"""Visual style of the paper figures, matching figures_model_description/: bold panel letter and title, grey units
subtitle, light horizontal grid, no box, one blue for the main series and orange for the contrast."""
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

INK = '#1f1f1f'
TEXT = '#4d4d4d'
MUTED = '#9a9a9a'
GRID = '#e6e6e6'
ZERO = '#8c8c8c'
BAND = '#f1efec'
LAND = '#f4f2ef'

BLUE = '#2b7bd9'
BLUE_LIGHT = '#bcd5f2'
ORANGE = '#e8692f'
ORANGE_LIGHT = '#f6c9b3'
NEUTRAL = '#e3ddd6'

# Share of the emissions cut that comes from innovation: 0 all contraction (orange), 1 all innovation (blue)
INNOVATION_CMAP = LinearSegmentedColormap.from_list('innovation', [ORANGE, NEUTRAL, BLUE])

FONTS = ['Source Sans 3', 'Source Sans Pro', 'Segoe UI', 'Helvetica', 'Arial', 'DejaVu Sans']


def apply():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': FONTS,
        'font.size': 8.5,
        'axes.edgecolor': ZERO,
        'axes.labelcolor': TEXT,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'xtick.color': TEXT,
        'ytick.color': TEXT,
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'xtick.major.pad': 4,
        'ytick.major.pad': 4,
        'legend.frameon': False,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
    })


def panel_title(ax, letter, title, subtitle=None):
    """Bold 'a  Title' above the axes, with the units in grey underneath."""
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(0, 24), textcoords='offset points',
                fontsize=10.5, fontweight='bold', color=INK, ha='left', va='bottom')
    ax.annotate(title, xy=(0, 1), xycoords='axes fraction', xytext=(14, 24), textcoords='offset points',
                fontsize=10.5, fontweight='bold', color=INK, ha='left', va='bottom')
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords='axes fraction', xytext=(0, 9), textcoords='offset points',
                    fontsize=8, color=MUTED, ha='left', va='bottom')


def zero_line(ax, axis='y'):
    (ax.axhline if axis == 'y' else ax.axvline)(0, color=ZERO, lw=0.9, zorder=1)


def end_label(ax, x, y, text, color=INK, dy=6, **kw):
    ax.annotate(text, xy=(x, y), xytext=(0, dy), textcoords='offset points', ha='center',
                va='bottom' if dy >= 0 else 'top', fontsize=8.5, fontweight='bold', color=color, **kw)


def spread(values, min_gap):
    """Positions for labels at `values`, moved apart by at least `min_gap`, keeping their order and mean."""
    order = np.argsort(values)
    pos = np.asarray(values, float)[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    pos += (np.asarray(values, float)[order].mean() - pos.mean())
    out = np.empty_like(pos)
    out[order] = pos
    return out


def fmt_pct(v, digits=0):
    v = round(float(v), digits) + 0.0  # no '-0%'
    return '{}{:.{}f}%'.format('+' if v > 0 else '−' if v < 0 else '', abs(v), digits)
