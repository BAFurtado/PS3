
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from analysis.output import OUTPUT_DATA_SPEC


def build_simulation_id(stats_path: Path) -> str:
    """
    Gera um ID único baseado no caminho completo da simulação
    """
    # pasta que contém INTEREST=...
    config_folder = next(
        p.name for p in stats_path.parents
        if '=' in p.name
    )

    # timestamp folder (um nível acima)
    timestamp_folder = stats_path.parents[
        [p.name for p in stats_path.parents].index(config_folder) + 1
    ].name

    sim_number = stats_path.parent.name  # 0,1,2

    return f"{timestamp_folder}__{config_folder}__{sim_number}"


def extract_metadata(stats_path: Path) -> dict:
    # parent of "0", i.e. the folder with INTEREST=... etc
    config_dir = next(
        p.name for p in stats_path.parents
        if '=' in p.name
    )
    parts = config_dir.split("__")
    meta = {}

    for part in parts:
        key, value = part.split("=")
        meta[key.lower()] = value

    # normalize types
    try:
        meta["policy_melhorias"] = meta["policy_melhorias"] == "True"
    except KeyError:
        pass

    return meta


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

        df = pd.read_csv(path, sep=';')
        df.columns = stats_cols
        meta = extract_metadata(path)
        for key, value in meta.items():
            df[key] = value
        df["simulation_id"] = (
                path.parents[2].name + "__" + path.parent.name
        )
        dfs.append(df)
    final = pd.concat(dfs, ignore_index=True)
    return final


if __name__ == '__main__':
    final_stats = main('stats')
    regional_stats = main('regional')

    final_stats.to_csv('final_stats.csv', index=False)
    regional_stats.to_csv('regional_stats.csv', index=False)

    # out = final_stats.groupby(by=['processing_acps', 'funds_availability', 'policy_melhorias'], as_index=False)[
    #     ['pop', 'price_level', 'gdp_level',
    #      'unemployment', 'firms_median_employment', 'families_median_wealth',
    #      'gini_index',
    #      'pct_zero_consumption', 'rent_default', 'inflation',
    #      'average_qli', 'house_vacancy', 'house_price', 'house_rent', "house_quality",
    #      'affordable', 'p_delinquent', ]].median(numeric_only=True)



