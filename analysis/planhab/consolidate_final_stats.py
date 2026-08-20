"""
consolidate_final_stats.py
===========================
Merges several already-built final_stats-style CSVs (each produced by a
separate batch/runner -- e.g. planhabanalysis.py output for one scenario
sweep) into a single consolidated database, one column per real-world
quantity even where a source's file is missing a column outright, or a
handful of rows within an otherwise-fine source lack it. This happens
because a batch's metadata-extraction step (planhabanalysis.py's
_CONF_SUPPLEMENT_KEYS) only captures swept parameters: e.g. the funds/
PLANHABFUNDS batch varies FUNDS_AVAILABILITY x POLICY_MELHORIAS but every
run in it still used a real (fixed) INTEREST_HOUSING value -- the
conf/default/params.py default ("media"), since main.py's PLANHABFUNDS
branch never overrides it. That's a known constant, not a missing
observation, so KNOWN_CONSTANTS below fills any remaining NaN with it
row-by-row (pandas .fillna, not a blanket overwrite) -- important for
downstream econometric use of the consolidated file, where a NaN control
would misread as "unknown" and silently drop those rows from any model
that conditions on it.

Row-wise (not source-wise) filling matters because a source's file isn't
guaranteed to be scenario-pure: e.g. final_stats_planhab_exclude_sp.csv
can pick up a few PLANHABFUNDS-shaped SAO PAULO rows (real
funds_availability, NaN interest_housing) while that -exclude_sp SAO PAULO
batch is still running on a separate --only-sp pass -- SAO PAULO takes far
longer than any other capital, so its raw output can already be on disk,
and get vacuumed up by whatever rglob-based script last rebuilt the CSV,
before the batch finishes. Those rows are legitimate (if
temporarily incomplete -- fewer than 360 months per simulation_id until
the run catches up) and are kept and filled exactly like every other row,
not dropped.

Built for scale: every source is read with pd.read_csv(chunksize=...) and
written incrementally with a streaming pyarrow ParquetWriter, so peak
memory stays bounded by CHUNK_ROWS regardless of how many total rows the
combined database ends up with (currently ~2.4M rows across the three
'final_stats' sources; designed to keep working if that grows toward the
tens of millions once more scenario batches / SAO PAULO runs land).

Run from the project root:
    /media/furtado/arthur/conda/envs/ps3_2/bin/python \\
        analysis/planhab/consolidate_final_stats.py --base final_stats

To add a new source (a new runner's output, or the regional_stats
counterpart once regional_stats_*.csv siblings exist), just add an entry
to SOURCES below -- nothing else needs to change.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# We immediately astype() every filled column to reference_dtypes anyway, so
# fillna's own dtype-inference choice doesn't matter -- opt in to the future
# behavior to silence the FutureWarning pandas raises otherwise.
pd.set_option('future.no_silent_downcasting', True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / 'output'

CHUNK_ROWS = 200_000

# base -> {source_label: filename (relative to output/)}.
# Every file under one base is expected to share (a superset/subset of) the
# same columns -- e.g. all are final_stats-shaped, one row per
# simulation_id x month. Extend this dict to add new batches.
SOURCES = {
    'final_stats': {
        'planhab':           'final_stats_planhab_exclude_sp.csv',
        'funds':             'final_stats_funds_exclude_sp.csv',
        'density_submitted': 'old/final_stats_agglomeration_submitted.csv',
    },
    'regional_stats': {
        # e.g. 'planhab': 'regional_stats_planhab_exclude_sp.csv',
        #      'funds':   'regional_stats_funds_exclude_sp.csv',
    },
}

# base -> {column: known_constant_value}.
# Params that are fixed across every run currently registered for this base,
# whether the column is entirely absent from some/all sources (e.g.
# policy_mcmv, never captured by planhabanalysis.py's _CONF_SUPPLEMENT_KEYS
# since nothing sweeps it yet) or present but NaN for a handful of stray
# rows within an otherwise-fine source (e.g. interest_housing for the
# SAO PAULO rows described in the module docstring). Applied with
# .fillna(), row by row, AFTER reindexing has ensured the column exists --
# so real varying values already present anywhere are never touched, and a
# future batch that genuinely varies one of these (e.g. an MCMV=False
# sweep) just needs a real per-row column of the same name; no schema
# migration needed on rows already written here. Parquet's columnar
# (dictionary/RLE) encoding makes a constant column cost next to nothing
# on disk even across millions of rows.
#
# interest_housing: conf/default/params.py default is "media"; the
# PLANHABFUNDS branch in main.py never overrides it.
# policy_mcmv: conf/default/params.py default is True; neither the PLANHAB
# nor PLANHABFUNDS branch overrides it, and neither runner script passes a
# -p PARAMS override, so it's True for every row registered below as of
# 2026-07-31.
KNOWN_CONSTANTS = {
    'final_stats': {
        'interest_housing': 'media',
        'policy_mcmv': True,
    },
}


def registered_sources(base: str) -> dict:
    sources = SOURCES.get(base)
    if not sources:
        raise ValueError(
            f"No sources registered for base={base!r}. "
            f"Known bases: {list(SOURCES)}. Add entries to SOURCES in this file."
        )
    missing = [f for f in sources.values() if not (OUTPUT_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing source file(s) under {OUTPUT_DIR}: {missing}"
        )
    return sources


def count_rows(path: Path) -> int:
    """Fast line count (data rows, excluding header) without loading the file."""
    n = 0
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            n += block.count(b'\n')
    return n - 1  # header line


def survey(base: str) -> None:
    """Print sizes/row counts up front so scale is known before writing anything."""
    sources = registered_sources(base)
    print(f"[{base}] sources:")
    total_rows = 0
    total_bytes = 0
    all_columns: list[str] = []
    for label, filename in sources.items():
        path = OUTPUT_DIR / filename
        n_rows = count_rows(path)
        n_bytes = path.stat().st_size
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        for c in cols:
            if c not in all_columns:
                all_columns.append(c)
        total_rows += n_rows
        total_bytes += n_bytes
        print(f"    {label:20s} {filename:40s} "
              f"{n_rows:>10,} rows  {n_bytes / 1e6:>8.1f} MB  {len(cols)} cols")
    known = KNOWN_CONSTANTS.get(base, {})
    all_columns_plus_known = all_columns + [c for c in known if c not in all_columns]
    print(f"    {'TOTAL':20s} {'':40s} {total_rows:>10,} rows  {total_bytes / 1e6:>8.1f} MB")
    print(f"    union of columns across sources: {len(all_columns)}"
          f"{f' (+{len(known)} known-constant)' if known else ''}")

    for label, filename in sources.items():
        cols = set(pd.read_csv(OUTPUT_DIR / filename, nrows=0).columns)
        missing = [c for c in all_columns_plus_known if c not in cols]
        for col in missing:
            if col in known:
                print(f"    [{label}] column {col!r} absent from this file -> filled with {known[col]!r} for every row")
            else:
                print(f"    [{label}] column {col!r} absent from this file -> WILL BE LEFT NaN "
                      f"(add to KNOWN_CONSTANTS if this was actually held constant)")
        present_with_gaps = [c for c in known if c in cols]
        if present_with_gaps:
            print(f"    [{label}] present but will fillna() any remaining gaps in: {present_with_gaps} "
                  f"(e.g. stray rows from another in-flight scenario -- see module docstring)")


def consolidate(base: str, out_name: str | None = None, chunk_rows: int = CHUNK_ROWS) -> Path:
    sources = registered_sources(base)
    out_path = OUTPUT_DIR / (out_name or f'{base}_consolidated.parquet')

    # Fix the schema before writing any data: union of columns across all
    # sources, plus a 'source' column recording which batch each row came
    # from (more reliable than the pre-existing 'run_type' column, whose
    # meaning differs across these files -- see notes in planhabanalysis.py's
    # infer_run_type).
    canonical_columns: list[str] = []
    for filename in sources.values():
        for c in pd.read_csv(OUTPUT_DIR / filename, nrows=0).columns:
            if c not in canonical_columns:
                canonical_columns.append(c)

    known = KNOWN_CONSTANTS.get(base, {})
    # Columns absent from every current source's CSV (e.g. policy_mcmv)
    # aren't picked up by the loop above -- add them to the schema explicitly.
    for col in known:
        if col not in canonical_columns:
            canonical_columns.append(col)

    # Canonical dtypes: sniffed from whichever source has every column
    # (falls back column-by-column across sources for anything still missing).
    reference_dtypes: dict = {}
    for filename in sources.values():
        sample = pd.read_csv(OUTPUT_DIR / filename, nrows=2000, low_memory=False)
        for c in sample.columns:
            if c not in reference_dtypes:
                reference_dtypes[c] = sample[c].dtype
    for col, value in known.items():
        reference_dtypes.setdefault(col, pd.Series([value]).dtype)

    canonical_columns.append('source')

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    rows_by_source: dict = {}
    filled_counts: dict = {}  # column -> total rows where a NaN was fillna()'d in
    t0 = time.time()

    try:
        for label, filename in sources.items():
            path = OUTPUT_DIR / filename
            print(f"[{label}] reading {filename} in chunks of {chunk_rows:,} rows ...")
            rows_by_source[label] = 0

            for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
                chunk = chunk.reindex(columns=[c for c in canonical_columns if c != 'source'])
                # Row-wise fillna (not a blanket overwrite): a column already
                # populated with real values anywhere in this chunk keeps
                # them; only genuine gaps -- whether the whole column was
                # absent from this source, or a handful of stray rows within
                # an otherwise-fine source (see module docstring) -- get the
                # known constant.
                for col, value in known.items():
                    na_mask = chunk[col].isna()
                    if na_mask.any():
                        filled_counts[col] = filled_counts.get(col, 0) + int(na_mask.sum())
                        chunk[col] = chunk[col].fillna(value)
                for col, dtype in reference_dtypes.items():
                    if chunk[col].dtype != dtype:
                        chunk[col] = chunk[col].astype(dtype)
                chunk['source'] = label

                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)

                total_rows += len(chunk)
                rows_by_source[label] += len(chunk)
                elapsed = time.time() - t0
                print(f"    ... {total_rows:,} rows written total  ({elapsed:.0f}s)", end='\r')
            print()
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - t0
    out_bytes = out_path.stat().st_size
    print(f"\nDone in {elapsed:.0f}s -- wrote {total_rows:,} rows "
          f"({len(canonical_columns)} cols) to {out_path}")
    print(f"  size on disk: {out_bytes / 1e6:.1f} MB")
    for label, n in rows_by_source.items():
        print(f"  {label:20s} {n:>10,} rows")
    for col, n in filled_counts.items():
        print(f"  filled {n:,} NaN in {col!r} with {known[col]!r}")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate multiple same-structure output CSVs into one Parquet database."
    )
    parser.add_argument(
        "--base", default="final_stats", choices=list(SOURCES),
        help="Which registered source group to consolidate (default: final_stats).",
    )
    parser.add_argument(
        "--out-name", default=None,
        help="Output filename under output/ (default: <base>_consolidated.parquet).",
    )
    parser.add_argument(
        "--survey-only", action="store_true",
        help="Only print source sizes/row counts, don't write the consolidated file.",
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=CHUNK_ROWS,
        help=f"Rows per chunk for reading/writing (default: {CHUNK_ROWS:,}).",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    survey(args.base)
    if not args.survey_only:
        consolidate(args.base, out_name=args.out_name, chunk_rows=args.chunk_rows)