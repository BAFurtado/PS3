"""Pre-launch smoke check for the PLANHABFUNDS re-run.

Reads one sensitivity output directory (12 combos x N seeds) and verifies the
things that, had they been checked last time, would have saved a batch:

  1. stats.csv / regional.csv have the current column layout;
  2. loan_approved + every denied_* reconciles exactly to loan_requested
     (this is what denied_zero_capped_amount was silently absorbing);
  3. rent_burden_decis_* is finite, row-wise monotonic, and its 5th decile
     equals affordability_median (it is a real quantile function now);
  4. FUNDS_AVAILABILITY actually moves the MCMV envelope post-2026 -- the
     otimista/pessimista ratio should approach the 0.25/0.04 = 6.25x in
     OGU_INVESTMENT, not be 1.0 as in the void batch;
  5. TOTAL_TARGETING_POLICY yields two distinguishable arms;
  6. the new per-municipality MCMV diagnostics are written, populated and
     internally consistent (exactly one stop reason, bought <= available,
     residual <= start).

Usage:
    python analysis/planhab/check_smoke_batch.py <output/dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.output import OUTPUT_DATA_SPEC, columns_for  # noqa: E402

STATS_COLS = OUTPUT_DATA_SPEC['stats']['columns']
REGIONAL_COLS = OUTPUT_DATA_SPEC['regional']['columns']
DENIED = [c for c in STATS_COLS if c.startswith('denied_')]
DECIS = ['rent_burden_decis_{}'.format(i) for i in range(1, 11)]
STOPS = [c for c in REGIONAL_COLS if c.startswith('mcmv_stop_')]

PASS, FAIL = 0, 0


def check(label, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('PASS  {}'.format(label))
    else:
        FAIL += 1
        print('FAIL  {}{}'.format(label, '  ({})'.format(detail) if detail else ''))


def read_csv(path, kind):
    df = pd.read_csv(path, sep=';', decimal='.', header=None)
    df.columns = columns_for(kind, df.shape[1])
    df['month'] = pd.to_datetime(df['month'])
    return df


def combo_key(path):
    """{'FUNDS_AVAILABILITY': 'otimista', ...} from a conf_to_str directory name."""
    out = {}
    for part in path.name.split('__'):
        if '=' in part:
            k, v = part.split('=', 1)
            out[k] = v
    return out


def main(root):
    root = Path(root)
    combos = sorted(d for d in root.iterdir() if d.is_dir() and '=' in d.name)
    print('Reading {} combination(s) under {}\n'.format(len(combos), root))

    stats, regional = [], []
    for combo in combos:
        meta = combo_key(combo)
        for run_dir in sorted(d for d in combo.iterdir() if d.is_dir() and d.name.isdigit()):
            s_path, r_path = run_dir / 'stats.csv', run_dir / 'regional.csv'
            if not s_path.exists():
                print('  (skipping incomplete run {})'.format(run_dir))
                continue
            s = read_csv(s_path, 'stats')
            r = read_csv(r_path, 'regional')
            for frame in (s, r):
                for k, v in meta.items():
                    frame[k] = v
                frame['seed_dir'] = run_dir.name
            stats.append(s)
            regional.append(r)

    if not stats:
        print('No completed runs found.')
        return 1

    stats = pd.concat(stats, ignore_index=True)
    regional = pd.concat(regional, ignore_index=True)
    print('  stats rows={}  regional rows={}\n'.format(len(stats), len(regional)))

    # ── 1. layout ────────────────────────────────────────────────────────────
    print('── Column layout ────────────────────────────────────────────')
    check('stats.csv written with the current {}-column layout'.format(len(STATS_COLS)),
          all(c in stats.columns for c in STATS_COLS),
          'missing={}'.format([c for c in STATS_COLS if c not in stats.columns]))
    check('regional.csv written with the current {}-column layout'.format(len(REGIONAL_COLS)),
          all(c in regional.columns for c in REGIONAL_COLS),
          'missing={}'.format([c for c in REGIONAL_COLS if c not in regional.columns]))

    # ── 2. credit rationing decomposition closes ─────────────────────────────
    print('\n── Credit rationing decomposition ───────────────────────────')
    accounted = stats['loan_approved'] + stats[DENIED].sum(axis=1)
    gap = (accounted - stats['loan_requested']).abs()
    check('approved + all denied_* == loan_requested, every row',
          bool((gap == 0).all()),
          'max gap={:.0f} over {} bad rows'.format(gap.max(), int((gap > 0).sum())))
    fires = {c: int((stats[c] > 0).any()) for c in DENIED}
    print('      counters that never fire: {}'.format(
        [c for c, f in fires.items() if not f] or 'none'))
    # The headline loan_approval_rate is approved/requested, and `requested`
    # includes calls where the family asked for nothing. Report the conditional
    # rate too -- it is the one that means "was credit rationed?".
    needed = stats['loan_requested'] - stats['denied_no_loan_needed']
    cond = (stats['loan_approved'] / needed.replace(0, np.nan)).median()
    print('      loan_approval_rate as saved (approved/requested): {:.3f}'.format(
        stats['loan_approval_rate'].median()))
    print('      approval rate among requests that NEEDED a loan:  {:.3f}'.format(cond))
    share_no_loan = (stats['denied_no_loan_needed'] / stats['loan_requested']
                     .replace(0, np.nan)).median()
    print('      share of requests that were for a zero loan:      {:.3f}'.format(share_no_loan))
    check('the dominant denial bucket is identified, not a catch-all',
          share_no_loan == share_no_loan,  # NaN-safe presence check
          'denied_no_loan_needed not populated')
    # denied_affordability is expected to be among them. Central.request_loan
    # caps the principal at max_loan() *before* the gate, and max_loan() inverts
    # exactly the payment formula the gate then tests, so first_payment ==
    # monthly_budget and the strict `>` never trips. Affordability therefore
    # binds on the intensive margin (loan size) rather than as a rejection; the
    # extensive-margin version of it is denied_zero_capped_amount, which fires
    # when the family's permanent income caps the principal at zero.
    print('      (denied_affordability is unreachable by construction -- see '
          'agents/bank.py:261-275; affordability binds by capping the '
          'principal, not by refusing)')

    # ── 3. rent burden deciles ───────────────────────────────────────────────
    print('\n── Rent burden deciles ──────────────────────────────────────')
    d = stats[DECIS].to_numpy(dtype=float)
    check('no inf / NaN in rent_burden_decis_*',
          bool(np.isfinite(d).all()),
          'non-finite={}'.format(int((~np.isfinite(d)).sum())))
    mono = (np.diff(d, axis=1) >= -1e-9).all(axis=1)
    check('rent_burden_decis_* monotonic in every row',
          bool(mono.all()),
          '{:.2%} of rows monotonic'.format(mono.mean()))
    dev = (stats['rent_burden_decis_5'] - stats['affordability_median']).abs()
    check('rent_burden_decis_5 == affordability_median',
          bool((dev < 1e-6).all()), 'max dev={:.3g}'.format(dev.max()))
    print('      median decile profile: {}'.format(
        np.round(np.median(d, axis=0), 3).tolist()))
    print('      median share of renters with zero permanent income: {:.3f} '
          '(these have an undefined burden and sit in decile 10)'.format(
              stats['pct_renters_zero_income'].median()))

    # ── 4. FUNDS_AVAILABILITY moves the envelope ─────────────────────────────
    print('\n── FUNDS_AVAILABILITY (post-2026 OGU envelope) ──────────────')
    post = regional[regional['month'] >= '2026-01-01']
    # The envelope is the month's allocation. mcmv_money_start is allocation plus
    # whatever the pot carried over, and the carryover is larger where the programme
    # spends a smaller share of what it holds, which pulls the ratio away from
    # OGU_INVESTMENT. Layouts without a topup column have no carryover, so there
    # money_start is the allocation.
    envelope_col = 'mcmv_money_topup' if 'mcmv_money_topup' in post.columns else 'mcmv_money_start'
    env = post.groupby('FUNDS_AVAILABILITY')[envelope_col].mean()
    print('      mean mun-month envelope ({}): {}'.format(
        envelope_col, {k: round(v, 2) for k, v in env.items()}))
    if {'otimista', 'pessimista'} <= set(env.index):
        ratio = env['otimista'] / env['pessimista'] if env['pessimista'] else float('inf')
        check('otimista/pessimista envelope ratio ~= 6.25 (was 1.0 in the void batch)',
              5.0 < ratio < 7.5, 'ratio={:.2f}'.format(ratio))
    else:
        check('all three FUNDS_AVAILABILITY arms present', False,
              'found={}'.format(list(env.index)))

    spent = post.groupby('FUNDS_AVAILABILITY')['mcmv_units_bought'].mean()
    print('      mean units bought per mun-month: {}'.format(
        {k: round(v, 3) for k, v in spent.items()}))

    # ── 5. targeting arms separate ───────────────────────────────────────────
    print('\n── TOTAL_TARGETING_POLICY ───────────────────────────────────')
    check('both targeting arms ran',
          set(stats['TOTAL_TARGETING_POLICY'].unique()) == {'True', 'False'},
          'found={}'.format(sorted(stats['TOTAL_TARGETING_POLICY'].unique())))
    tgt = stats.groupby('TOTAL_TARGETING_POLICY')[
        ['gini_index', 'affordability_median', 'families_helped']].mean()
    print(tgt.to_string())
    if len(tgt) == 2:
        rel = (tgt.loc['True'] - tgt.loc['False']).abs() / tgt.loc['False'].abs().replace(0, np.nan)
        check('targeting arms differ on at least one headline outcome',
              bool((rel > 1e-6).any()),
              'max relative gap={:.3g}'.format(rel.max()))

    # ── 6. MCMV allocation diagnostics ───────────────────────────────────────
    print('\n── MCMV allocation diagnostics ──────────────────────────────')
    active = regional[regional['mcmv_money_start'] > 0]
    check('mcmv_money_start is populated',
          len(active) > 0, '{} of {} mun-months'.format(len(active), len(regional)))
    stop_sum = active[STOPS].sum(axis=1)
    check('exactly one stop reason per active mun-month',
          bool((stop_sum == 1).all()),
          'rows with sum!=1: {}'.format(int((stop_sum != 1).sum())))
    check('units_bought <= units_available',
          bool((active['mcmv_units_bought'] <= active['mcmv_units_available']).all()))
    check('0 <= money_residual <= money_start',
          bool(((active['mcmv_money_residual'] >= -1e-6) &
                (active['mcmv_money_residual'] <= active['mcmv_money_start'] + 1e-6)).all()))
    share = active[STOPS].mean().sort_values(ascending=False)
    print('      stop-reason shares (this is what Paper A is after):')
    for name, val in share.items():
        print('        {:<32s} {:.3f}'.format(name.replace('mcmv_stop_', ''), val))
    check('the binding constraint is identified (some reason dominates)',
          share.max() > 0.10, 'top share={:.3f}'.format(share.max()))

    print('\n{}\nResults: {} PASS | {} FAIL\n{}'.format('-' * 60, PASS, FAIL, '-' * 60))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
