
import json
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from analysis.output import OUTPUT_DATA_SPEC

# Keys pulled from conf.json to supplement the folder-name metadata.
# These matter most for sensitivity runs where the folder name only
# contains the swept parameter and the city is buried in conf.json.
_CONF_SUPPLEMENT_KEYS = [
    'PROCESSING_ACPS',
    'INTEREST',
    'FUNDS_AVAILABILITY',
    'POLICY_MELHORIAS',
]

_PLANHAB_KEYS = {'processing_acps', 'policy_melhorias', 'funds_availability'}


def extract_metadata(stats_path: Path) -> dict:
    meta = {}

    # 1. Parse KEY=VALUE pairs from the config folder name.
    try:
        config_dir = next(p.name for p in stats_path.parents if '=' in p.name)
        for part in config_dir.split("__"):
            key, value = part.split("=", 1)
            meta[key.lower()] = value
    except StopIteration:
        pass  # plain run with no config folder (timestamp only)

    # 2. Fill in fields absent from the folder name using conf.json.
    conf_path = stats_path.parent / 'conf.json'
    if conf_path.exists():
        try:
            with open(conf_path) as f:
                run_conf = json.load(f)
            params = run_conf.get('PARAMS', {})
            for key in _CONF_SUPPLEMENT_KEYS:
                if key.lower() not in meta and key in params:
                    val = params[key]
                    if isinstance(val, list):
                        val = ','.join(str(v) for v in val)
                    meta[key.lower()] = val
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    # Normalize boolean
    if 'policy_melhorias' in meta:
        meta['policy_melhorias'] = meta['policy_melhorias'] in (True, 'True')

    return meta


def _folder_meta(stats_path: Path) -> dict:
    """Metadata from the config folder name only — no conf.json supplement."""
    try:
        config_dir = next(p.name for p in stats_path.parents if '=' in p.name)
        meta = {}
        for part in config_dir.split("__"):
            k, v = part.split("=", 1)
            meta[k.lower()] = v
        return meta
    except StopIteration:
        return {}


def infer_run_type(stats_path: Path) -> str:
    folder_keys = set(_folder_meta(stats_path).keys())
    if {'policy_melhorias', 'funds_availability'}.issubset(folder_keys):
        return 'planhab'
    non_city = folder_keys - {'processing_acps', 'interest'}
    if len(non_city) == 1:
        return 'sensitivity'
    return 'other'


def main(base='stats'):
    HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = HERE.parents[1]

    path_base = PROJECT_ROOT / 'output'

    stats_files = [
        p for p in path_base.rglob(f'{base}.csv')
        if p.parent.name != 'avg'
    ]
    stats_cols = OUTPUT_DATA_SPEC[base]['columns']

    dfs = []
    for path in stats_files:
        try:
            df = pd.read_csv(path, sep=';')
            df.columns = stats_cols
        except (pd.errors.ParserError, ValueError, OSError):
            continue

        meta = extract_metadata(path)
        for key, value in meta.items():
            df[key] = value

        df['run_type'] = infer_run_type(path)
        # unique ID: timestamp__config__run_number
        df['simulation_id'] = (
            path.parents[2].name + '__' + path.parents[1].name + '__' + path.parent.name
        )
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    final_stats = main('stats')
    regional_stats = main('regional')

    final_stats.to_csv('final_stats.csv', index=False)
    regional_stats.to_csv('regional_stats.csv', index=False)

    print(f"Saved {len(final_stats)} rows to final_stats.csv")
    print(f"Saved {len(regional_stats)} rows to regional_stats.csv")
    if not final_stats.empty:
        print(final_stats.groupby('run_type')['simulation_id'].nunique().rename('unique_runs').to_string())
