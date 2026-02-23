#!/usr/bin/env python3

from pathlib import Path
import json
import csv
import argparse
from collections import defaultdict


# =========================
# CONFIGURATION
# =========================

ALLOWED_TO_VARY = {
    "PROCESSING_ACPS",
    "POLICY_MCMV",
    "POLICY_MELHORIAS",
    "INTEREST",
    "SEED"
}


# =========================
# UTILITIES
# =========================

def normalize(value):
    """
    Normalize values so comparisons are robust.
    """
    if isinstance(value, float):
        return round(value, 8)

    if isinstance(value, list):
        return tuple(sorted(normalize(v) for v in value))

    if isinstance(value, dict):
        return tuple(sorted((k, normalize(v)) for k, v in value.items()))

    return value


def flatten_params(param_list):
    """
    Flatten a list of dictionaries (possibly nested lists).
    """
    flat = {}

    def _flatten(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flat[k] = normalize(v)
        elif isinstance(obj, list):
            for item in obj:
                _flatten(item)

    _flatten(param_list)
    return flat


# =========================
# CORE FUNCTIONS
# =========================

def collect_meta_files(root_dir):
    root = Path(root_dir)
    return list(root.rglob("meta.json"))


def extract_params(meta_path):
    with open(meta_path, "r") as f:
        data = json.load(f)

    all_params = []

    for entry in data:
        params = entry.get("params", [])
        all_params.append(params)

    return all_params


def collect_all_runs(root_dir):
    meta_files = collect_meta_files(root_dir)

    runs = []

    for meta_file in meta_files:
        extracted = extract_params(meta_file)

        for param_list in extracted:
            flat = flatten_params(param_list)

            runs.append({
                "file": str(meta_file),
                "params": flat
            })

    return runs


def analyze_variations(runs):
    values_per_key = defaultdict(set)

    for run in runs:
        for k, v in run["params"].items():
            values_per_key[k].add(v)

    inconsistent = {}

    for k, values in values_per_key.items():
        if len(values) > 1 and k not in ALLOWED_TO_VARY:
            inconsistent[k] = values

    return values_per_key, inconsistent


# =========================
# CSV EXPORT
# =========================

def export_all_values(values_per_key, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Values Found"])

        for k in sorted(values_per_key):
            writer.writerow([k, list(values_per_key[k])])


def export_inconsistencies(inconsistent, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Unexpected Values"])

        for k in sorted(inconsistent):
            writer.writerow([k, list(inconsistent[k])])


def compare_with_expected(values_per_key, expected_csv, output_path):
    expected = {}

    with open(expected_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected[row["Parameter"]] = row["ExpectedValue"]

    mismatches = []

    for param, expected_value in expected.items():
        actual_values = values_per_key.get(param, None)

        if actual_values is None:
            mismatches.append((param, "MISSING", expected_value))
        elif len(actual_values) != 1 or str(next(iter(actual_values))) != expected_value:
            mismatches.append((param, list(actual_values), expected_value))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Actual Values", "Expected Value"])
        writer.writerows(mismatches)


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Verify consistency of simulation parameters."
    )

    parser.add_argument("root", help="Root directory containing simulation folders")
    parser.add_argument("--expected", help="CSV file with expected parameters")
    parser.add_argument("--outdir", default="verification_output",
                        help="Output directory for reports")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print("Collecting simulation runs...")
    runs = collect_all_runs(args.root)

    print(f"Total runs found: {len(runs)}")

    values_per_key, inconsistent = analyze_variations(runs)

    print("\n=== Parameter Summary ===")
    for k in sorted(values_per_key):
        print(f"{k}: {values_per_key[k]}")

    print("\n=== Unexpected Variations ===")
    if not inconsistent:
        print("No unexpected variations found.")
    else:
        for k, v in inconsistent.items():
            print(f"{k}: {v}")

    export_all_values(values_per_key, outdir / "all_parameter_values.csv")
    export_inconsistencies(inconsistent, outdir / "unexpected_variations.csv")

    if args.expected:
        compare_with_expected(
            values_per_key,
            args.expected,
            outdir / "comparison_with_expected.csv"
        )
        print("\nComparison with expected table completed.")

    print(f"\nReports saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()