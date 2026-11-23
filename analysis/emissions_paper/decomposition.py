"""LMDI split of an emissions change between two scenarios into intensity (innovation) and scale (contraction).

A firm emits E = eff * a_s * B / EMISSIONS_PARAM (Firm.create_externalities): eff its environmental efficiency, a_s its
sector's intensity, B the money base of that month (revenue deflated by the sector's own accumulated inflation with
EMISSIONS_BASE 'sector_real_output', by the economy-wide one with 'real_output', nominal revenue with 'output'). Summed over a sector,
E_s = a_s / EMISSIONS_PARAM * B_s * e_s with e_s = sum(eff * B) / B_s, the base-weighted efficiency. Between a
treated (T) and a reference (N) run, additive LMDI-I (Ang 2005) gives, sector by sector,
    intensity = L(E_T, E_N) * ln(e_T / e_N)      falls only when firms innovate
    scale     = L(E_T, E_N) * ln(B_T / B_N)      falls when the sector produces less
with L the logarithmic mean, and intensity + scale = E_T - E_N exactly. A sector present in one run only goes
wholly to scale. Shifts between sectors land in scale too: they are contraction of the emitting sectors.
"""
import numpy as np
import pandas as pd

KEYS = ['city', 'seed', 'year', 'sector']


def logmean(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (a - b) / (np.log(a) - np.log(b))
    return np.where(np.isclose(a, b), a, out)


def lmdi(treated, reference):
    """Sector-level effects. Both frames: KEYS + E (emissions), B (base), e (base-weighted efficiency).
    Returns KEYS + E_T, E_N, intensity, scale, on the rows present in either frame."""
    m = treated[KEYS + ['E', 'B', 'e']].merge(reference[KEYS + ['E', 'B', 'e']], on=KEYS, how='outer',
                                               suffixes=('_T', '_N')).fillna(0.0)
    both = (m.E_T > 0) & (m.E_N > 0)
    L = np.where(both, logmean(m.E_T.where(both, 1), m.E_N.where(both, 1)), 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        m['intensity'] = np.where(both, L * np.log(m.e_T.where(both, 1) / m.e_N.where(both, 1)), 0.0)
        m['scale'] = np.where(both, L * np.log(m.B_T.where(both, 1) / m.B_N.where(both, 1)), m.E_T - m.E_N)
    return m[KEYS + ['E_T', 'E_N', 'intensity', 'scale']]


def innovation_share(intensity, scale):
    """Share of the cut that comes from innovation: the reducing parts only, so a sector that grows does not
    offset another that innovates. NaN where nothing falls."""
    cut_i, cut_s = np.maximum(-np.asarray(intensity), 0), np.maximum(-np.asarray(scale), 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(cut_i + cut_s > 0, cut_i / (cut_i + cut_s), np.nan)
