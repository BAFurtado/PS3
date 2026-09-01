"""
Fetch real-world empirical data for Belo Horizonte to use as calibration
targets (replaces the hardcoded placeholder OBSERVED dict in sample.py).

Sources (all via IBGE SIDRA API, apisidra.ibge.gov.br):
  - Inflation (IPCA, monthly): BH metro region, N7=3101. Spliced across three
    tables because IBGE changed the IPCA basket/methodology twice in this
    window (2006-2011 / 2012-2019 / 2020-present each have their own table).
  - Unemployment (PNADC, quarterly): BH metro region, N7=3101, from 2012
    onward. Pre-2012 (PME) is deliberately NOT fetched here: PME's SIDRA
    tables only expose unemployment split by family role (household head vs.
    other members), not a pooled total, and calibration's fitness window is
    meant to exclude burn-in (pre-2012) anyway per calibration_conf.py's
    burn_in_end. Revisit only if the R1 manuscript validation figure wants
    a full 2010+ empirical overlay during burn-in.
  - GDP level + growth (PIB dos Municípios, annual): summed across every
    municipality in the Belo Horizonte ACP, not just the core municipality
    (3106200) alone, since the model treats the whole metro region as one
    economy.

Municipality set for the ACP-level GDP aggregation comes from
input/ACPs_MUN_CODES.csv (columns: ACPs;cod_mun).

Usage:
    python -m analysis.calibration.fetch_municipal_validation_data
"""
import csv
import json
import statistics
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACP_MUN_CODES_PATH = ROOT / "input" / "ACPs_MUN_CODES.csv"
OUT_DIR = ROOT / "analysis" / "calibration" / "data"

ACP_NAME = "BELO HORIZONTE"
BH_METRO_N7 = "3101"  # SIDRA "Região Metropolitana até 2020" code for Belo Horizonte

# (table_id, variable_id, first_period, last_period) - last_period=None means "through latest"
IPCA_TABLES = [
    ("2938", "63", 201001, 201112),
    ("1419", "63", 201201, 201912),
    ("7060", "63", 202001, 202607),  # matches table 7060's published period range
]
IPCA_CLASSIF = "c315/7169"  # "Índice geral" (headline index, not a sub-item)

PNADC_UNEMPLOYMENT_TABLE = "4093"
PNADC_UNEMPLOYMENT_VAR = "4099"
PNADC_START_PERIOD = 201201

PIB_MUNICIPAL_TABLE = "5938"
PIB_MUNICIPAL_VAR = "37"  # "Produto Interno Bruto a preços correntes" (Mil Reais)

MISSING_MARKERS = {"...", "-", "X", ".."}


def _fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def load_bh_municipality_codes() -> list[str]:
    """Read the Belo Horizonte ACP's municipality codes from input/ACPs_MUN_CODES.csv."""
    codes = []
    with open(ACP_MUN_CODES_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if row["ACPs"].strip() == ACP_NAME:
                codes.append(row["cod_mun"].strip())
    if not codes:
        raise ValueError(f"No municipalities found for ACP '{ACP_NAME}' in {ACP_MUN_CODES_PATH}")
    return codes


def fetch_inflation() -> dict:
    """Monthly IPCA variation (%) for BH metro region, 2010-present."""
    series = {}
    for table_id, var_id, start, end in IPCA_TABLES:
        period = str(start) if end is None else f"{start}-{end}"
        url = (f"https://apisidra.ibge.gov.br/values/t/{table_id}/n7/{BH_METRO_N7}"
               f"/v/{var_id}/p/{period}/{IPCA_CLASSIF}")
        for row in _fetch_json(url)[1:]:
            month, val = row["D3C"], row["V"]
            if month >= "201001" and val not in MISSING_MARKERS:
                series[month] = float(val) / 100  # % -> fraction
    return dict(sorted(series.items()))


def fetch_unemployment() -> dict:
    """Quarterly unemployment rate (%) for BH metro region, 2012-present."""
    url = (f"https://apisidra.ibge.gov.br/values/t/{PNADC_UNEMPLOYMENT_TABLE}/n7/{BH_METRO_N7}"
           f"/v/{PNADC_UNEMPLOYMENT_VAR}/p/all")
    series = {}
    for row in _fetch_json(url)[1:]:
        quarter, val = row["D3C"], row["V"]
        if val not in MISSING_MARKERS:
            series[quarter] = float(val) / 100  # % -> fraction
    return dict(sorted(series.items()))


def fetch_gdp_level() -> dict:
    """Annual nominal GDP (thousand R$), summed across all BH ACP municipalities."""
    codes = load_bh_municipality_codes()
    totals: dict[str, float] = {}
    for code in codes:
        url = f"https://apisidra.ibge.gov.br/values/t/{PIB_MUNICIPAL_TABLE}/n6/{code}/v/{PIB_MUNICIPAL_VAR}/p/all"
        for row in _fetch_json(url)[1:]:
            year, val = row["D3C"], row["V"]
            if val not in MISSING_MARKERS:
                totals[year] = totals.get(year, 0.0) + float(val)
    return dict(sorted(totals.items()))


def gdp_growth_from_levels(levels: dict) -> dict:
    """Year-over-year real... (nominal, see note) growth rate from annual GDP levels.

    NOTE: PIB Municipal is nominal (current prices). A real-terms comparison
    would need an IPCA deflator applied before differencing; left as nominal
    for now since the model's own gdp_growth is not obviously real vs.
    nominal either (TODO: confirm against agents/firm.py's price-setting
    before treating this target as final).
    """
    years = sorted(levels)
    growth = {}
    for prev, cur in zip(years, years[1:]):
        if levels[prev]:
            growth[cur] = levels[cur] / levels[prev] - 1
    return growth


def summarize(series: dict, label: str):
    vals = list(series.values())
    if not vals:
        print(f"  {label}: NO DATA")
        return None
    mean, std = statistics.mean(vals), statistics.pstdev(vals)
    print(f"  {label}: n={len(vals)}, mean={mean:.5f}, std={std:.5f}, "
          f"range=[{min(series)}..{max(series)}]")
    return mean, std


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching BH-specific inflation (IPCA, monthly, 2010-present)...")
    inflation = fetch_inflation()

    print("Fetching BH-specific unemployment (PNADC, quarterly, 2012-present)...")
    unemployment = fetch_unemployment()

    print("Fetching BH ACP GDP level (PIB Municipal, annual, summed across "
          f"{len(load_bh_municipality_codes())} municipalities)...")
    gdp_level = fetch_gdp_level()
    gdp_growth = gdp_growth_from_levels(gdp_level)

    for name, series in [("inflation", inflation), ("unemployment", unemployment),
                          ("gdp_level", gdp_level), ("gdp_growth", gdp_growth)]:
        path = OUT_DIR / f"bh_{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["period", name])
            for k, v in series.items():
                w.writerow([k, v])
        print(f"  Saved: {path}")

    print("\n--- Summary (for calculate_fitness OBSERVED dict) ---")
    summarize(inflation, "inflation (monthly)")
    summarize(unemployment, "unemployment (quarterly)")
    summarize(gdp_growth, "gdp_growth (annual)")
    print("\nGini: NOT included (needs PNADC microdata processing - separate task).")


if __name__ == "__main__":
    main()
