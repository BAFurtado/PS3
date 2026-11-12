"""
Travel-time matrices between áreas de ponderação (APs) and their schedule over time.

Used by the labour market's commuting term (text_bndes / ENMU test). A matrix holds one column of
minutes per network *state*: 'base', 'nec' (the full ENMU Rede Necessária) and, for matrices we build
ourselves, any other named state (e.g. base plus a subset of projects).

Standard file, one per ACP: input/transport/travel_times_<ACP>.parquet, long format with columns
acp, ap_orig, ap_dest, state, minutes. Where no standard file exists, a partner file in its original
wide layout is read through PARTNER_FILES.

The schedule (PARAMS['TRANSPORT_SCHEDULE']) is a list of [date, spec] pairs, sorted by date, where date
is 'YYYY-MM-DD' or a year (int, meaning 1 January) and spec is either a state name or a number phi in
[0, 1], which blends minutes = (1 - phi) * base + phi * nec. A spec holds from its date until the next one.
Before the first date the network is 'base'. None keeps the old static switch: 'nec' when
PARAMS['TRANSPORT_TIME'] else 'base', for the whole run.
"""
import datetime
import os

import numpy as np
import pandas as pd

STANDARD_DIR = 'input/transport'
STANDARD_COLUMNS = ['acp', 'ap_orig', 'ap_dest', 'state', 'minutes']
CONGESTION = 'input/car_congestion_2022.csv'

# Partner files in their original layout: ACP -> (path, {column: state})
PARTNER_FILES = {
    'BRASILIA': ('input/bndes/travel_times_areapond_DF.parquet',
                 {'TempoRedeBase': 'base', 'TempoRedeNec': 'nec'}),
}


def standard_path(acp, weights=None):
    """With `weights` (TRANSPORT_WEIGHTS), the ENMU zone matrices converted with that weighting
    (tools/transport/partner_matrix.py)"""
    if weights:
        return os.path.join(STANDARD_DIR, 'travel_times_%s__%s.parquet' % (acp.replace(' ', '_'), weights))
    return os.path.join(STANDARD_DIR, 'travel_times_%s.parquet' % acp.replace(' ', '_'))


def read_acp_matrix(acp, weights=None):
    """Wide matrix for one ACP, indexed by (ap_orig, ap_dest), one column per state. None if absent."""
    path = standard_path(acp, weights)
    if weights and not os.path.exists(path):
        return None
    if os.path.exists(path):
        long = pd.read_parquet(path)
        missing = set(STANDARD_COLUMNS) - set(long.columns)
        if missing:
            raise ValueError('%s lacks columns %s' % (path, sorted(missing)))
        long = long[long['acp'] == acp]
        wide = long.pivot_table(index=['ap_orig', 'ap_dest'], columns='state', values='minutes', aggfunc='first')
        wide.columns.name = None
        return wide
    if acp in PARTNER_FILES:
        path, columns = PARTNER_FILES[acp]
        if not os.path.exists(path):
            return None
        raw = pd.read_parquet(path)
        wide = raw.rename(columns={'code_weighting_orig': 'ap_orig', 'code_weighting_dest': 'ap_dest', **columns})
        return wide.set_index(['ap_orig', 'ap_dest'])[list(columns.values())]
    return None


def to_standard(wide, acp):
    """Long standard table from a wide matrix, for writing to standard_path(acp)."""
    long = wide.reset_index().melt(id_vars=['ap_orig', 'ap_dest'], var_name='state', value_name='minutes')
    long.insert(0, 'acp', acp)
    return long.dropna(subset=['minutes'])[STANDARD_COLUMNS]


def check_matrix(wide, region_ids=None):
    """Problems that would make a matrix silently wrong. Empty list means it passed."""
    problems = []
    if 'base' not in wide.columns:
        problems.append("no 'base' state")
    values = wide.to_numpy()
    if (values < 0).any():
        problems.append('negative minutes')
    if pd.isna(values).any():
        problems.append('%d missing values' % int(pd.isna(values).sum()))
    if region_ids is not None:
        region_ids = set(region_ids)
        pairs = set(wide.index)
        absent = [(o, d) for o in region_ids for d in region_ids if (o, d) not in pairs]
        if absent:
            problems.append('%d of %d region pairs absent (e.g. %s)' % (len(absent), len(region_ids) ** 2, absent[0]))
    return problems


def check_nec_vs_base(wide, tolerance=.5):
    """Share of pairs where 'nec' is faster than 'base' by more than `tolerance` minutes, and mean change."""
    delta = wide['nec'] - wide['base']
    return {'share_faster': float((delta < -tolerance).mean()),
            'share_slower': float((delta > tolerance).mean()),
            'mean_change_min': float(delta.mean())}


def linear_phase_in(start_year, years):
    """Schedule that blends base into nec in equal yearly steps, fully open after `years` years."""
    return [[start_year + i, (i + 1) / years] for i in range(years)]


def _to_date(when):
    if isinstance(when, int):
        return datetime.date(when, 1, 1)
    return datetime.date.fromisoformat(when)


class TransportNetwork:
    """Travel-time matrix for the processing ACPs and the state in force on a given date."""

    def __init__(self, params, acps, logger=None):
        self.logger = logger
        matrices = {}
        weights = params.get('TRANSPORT_WEIGHTS')
        # The state that TRANSPORT_TIME switches to and a TRANSPORT_SCHEDULE phi blends toward
        self.future = params.get('TRANSPORT_FUTURE', 'nec')
        for acp in acps:
            wide = read_acp_matrix(acp, weights)
            if wide is not None:
                matrices[acp] = wide
        if params.get('CAR_CONGESTION', 'census') == 'census' and weights:
            # Free-flow car minutes times the ACP's congestion factor (auxiliary/car_congestion.py)
            kappa = pd.read_csv(CONGESTION, sep=';')
            kappa = kappa[kappa.weights == weights].drop_duplicates('acp', keep='last').set_index('acp').kappa
            for acp, wide in matrices.items():
                if 'car' in wide.columns:
                    if acp not in kappa.index:
                        raise ValueError('%s has no %r factor for %s' % (CONGESTION, weights, acp))
                    wide['car'] = wide['car'] * float(kappa[acp])
        self.matrix = None
        if not matrices:
            return
        if len(matrices) < len(acps):
            # Pairs outside a matrix would all get the fallback time, silently switching the
            # commuting term off there. Better to run every ACP on distance.
            self._log('warning', 'Travel-time matrix for %s but not %s: using distance for all ACPs'
                      % (sorted(matrices), sorted(set(acps) - set(matrices))))
            return
        self.matrix = pd.concat(matrices.values())
        self.acps = sorted(matrices)
        problems = check_matrix(self.matrix)
        if problems:
            raise ValueError('Travel-time matrix for %s: %s' % (self.acps, '; '.join(problems)))
        # State 'car': car minutes for car owners, the same in every network state
        self.car_time = None
        if 'car' in self.matrix.columns:
            self.car_time = self.matrix['car'].to_dict()
            self.max_car_time = float(self.matrix['car'].max())
        self.schedule = self._read_schedule(params)
        self.spec = None
        self.commute_time = None
        self.max_time = None
        self.cost_factor = None

    def check_coverage(self, region_ids):
        """Raise if some pair of simulated regions has no travel time."""
        problems = check_matrix(self.matrix, region_ids)
        if problems:
            raise ValueError('Travel-time matrix for %s: %s' % (self.acps, '; '.join(problems)))

    def calibrate_cost(self, regions, reg_pops):
        """Distance units per minute of the base network, so the transit cost parameters, calibrated on
        distance, charge the same average cost when applied to minutes. Ratio of centroid distance to base
        minutes over pairs of distinct regions, weighted by origin times destination population. It uses
        'base' whatever network is in force, so every scenario gets the same factor."""
        base = self.matrix['base']
        centroids = {r_id: r.addresses.centroid for r_id, r in regions.items()}
        dist_sum = min_sum = 0.
        for (o, d), minutes in base.items():
            if o == d or o not in centroids or d not in centroids:
                continue
            w = reg_pops[o] * reg_pops[d]
            dist_sum += w * centroids[o].distance(centroids[d])
            min_sum += w * minutes
        if min_sum <= 0:
            raise ValueError('Cannot calibrate transit cost: no weighted off-diagonal pairs')
        self.cost_factor = dist_sum / min_sum
        self._log('info', 'Transit cost factor: %.6g distance units per minute' % self.cost_factor)

    def minutes_between(self, orig, dest, car=False):
        if car and self.car_time is not None:
            return self.car_time.get((orig, dest), self.max_car_time)
        return self.commute_time.get((orig, dest), self.max_time)

    def _log(self, level, msg):
        if self.logger is not None:
            getattr(self.logger, level)(msg)
        else:
            print(msg)

    def _read_schedule(self, params):
        schedule = params.get('TRANSPORT_SCHEDULE')
        if schedule is None:
            state = self.future if params['TRANSPORT_TIME'] else 'base'
            return [(datetime.date.min, state)]
        schedule = sorted((_to_date(when), spec) for when, spec in schedule)
        for _, spec in schedule:
            if isinstance(spec, str):
                if spec not in self.matrix.columns:
                    raise ValueError('TRANSPORT_SCHEDULE state %r not in matrix states %s'
                                     % (spec, list(self.matrix.columns)))
            elif not 0 <= spec <= 1:
                raise ValueError('TRANSPORT_SCHEDULE phi %r outside [0, 1]' % spec)
            elif self.future not in self.matrix.columns:
                raise ValueError('TRANSPORT_SCHEDULE phi needs a %r state' % self.future)
        return schedule

    def spec_at(self, date):
        spec = 'base'
        for when, s in self.schedule:
            if when > date:
                break
            spec = s
        return spec

    def minutes(self, spec):
        if isinstance(spec, str):
            return self.matrix[spec]
        return (1 - spec) * self.matrix['base'] + spec * self.matrix[self.future]

    def update(self, date):
        """Set commute times for `date`. Returns True when the network changed."""
        spec = self.spec_at(date)
        if spec == self.spec:
            return False
        minutes = self.minutes(spec)
        self.commute_time = minutes.to_dict()
        self.max_time = float(minutes.max())
        self.spec = spec
        self._log('info', 'Transport network on %s: %s' % (date, spec))
        return True


ACCEPTANCE = 'input/commute_acceptance_2010.csv'


class CommuteAcceptance:
    """COMMUTE_ACCEPTANCE 'census': a candidate accepts an offer with probability P(t) / P(0),
    P(t) = 1 / (1 + exp((t - T50) / T_SCALE)), t its one-way minutes to the firm (car minutes for car owners). T50 and
    T_SCALE per ACP from input/commute_acceptance_2010.csv (auxiliary/commute_acceptance.py), the row of the run's
    TRANSPORT_WEIGHTS. The Census bands fix the shape of P, and P(0) = 1 fixes its level."""

    def __init__(self, acps, weights):
        table = pd.read_csv(ACCEPTANCE, sep=';')
        table = table[(table.weights == weights) & table.acp.isin(acps)].drop_duplicates('acp').set_index('acp')
        missing = set(acps) - set(table.index)
        if missing:
            raise ValueError('%s has no %r rows for %s' % (ACCEPTANCE, weights, sorted(missing)))
        self.params = {acp: (float(r.t50), float(r.t_scale)) for acp, r in table.iterrows()}
        muns = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
        muns = muns[muns.ACPs.isin(acps)]
        self.acp_by_mun = dict(zip(muns.cod_mun.astype(str), muns.ACPs))

    @staticmethod
    def logistic(t, t50, scale):
        return 1.0 / (1.0 + np.exp(np.clip((t - t50) / scale, -50.0, 50.0)))

    def probability(self, minutes, region_id):
        acp = self.acp_by_mun.get(str(region_id)[:7])
        if acp is None:
            return 1.0
        t50, scale = self.params[acp]
        return float(min(1.0, self.logistic(minutes, t50, scale) / self.logistic(0.0, t50, scale)))
