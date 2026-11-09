"""
Real-world emissions benchmark for a metro region (ACP): SEEG municipal CO2e
(GWP-100, as distributed in the SEEG workbook) summed over the ACP's
municipalities, and emissions per unit of real GDP.

Scopes (all "Emissão" rows; removals, NCI and bunker excluded):
  gross        every SEEG emission.
  firm_broad   gross minus household residential energy and land-use change.
               Upper bound on firm scope: still includes all road passenger
               transport (private cars + buses/taxis/ride-hailing).
  firm_narrow  firm_broad minus road passenger transport. Lower bound on firm
               scope: also drops the buses/taxis/vans that are firms.

Real GDP: PIB dos Municipios (SIDRA 5938, nominal, summed over the ACP's
municipalities) deflated to `--base-year` prices with the regional IPCA
(annual mean of the monthly price index). Intensity = tCO2e per R$ million of
base-year GDP.

The SEEG workbook is ~250 MB, so it is read once and aggregated to a small
per-ACP cache (data/seeg_acp_emissions.csv); later runs use the cache.

Usage (from the project root):
    python -m analysis.calibration.seeg_emissions
    python -m analysis.calibration.seeg_emissions --acp "SAO PAULO" --ipca-geo n7/3501
    python -m analysis.calibration.seeg_emissions --rebuild-cache
"""
import argparse
import csv
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.calibration.fetch_municipal_validation_data import (
    IPCA_CLASSIF,
    IPCA_TABLES,
    MISSING_MARKERS,
    PIB_MUNICIPAL_TABLE,
    PIB_MUNICIPAL_VAR,
    _fetch_json,
)

ROOT = Path(__file__).resolve().parents[2]
ACP_MUN_CODES_PATH = ROOT / "input" / "ACPs_MUN_CODES.csv"
XLSX_PATH = ROOT / "analysis" / "validation" / "real_world_data" / "emission_data_municipality.xlsx"
DATA_DIR = ROOT / "analysis" / "calibration" / "data"
RESULTS_DIR = ROOT / "analysis" / "validation" / "results"
CACHE_PATH = DATA_DIR / "seeg_acp_emissions.csv"

CACHE_MIN_YEAR = 2000
KEY_COLS = ["acp", "tipo", "setor", "cat", "sub", "ativ"]

# SEEG territory id = "1" + 7-digit IBGE municipality code.
SEEG_ID_PREFIX = "1"
# Positional layout of the SEEG "Dados" sheet (labels are accented, so by position).
COL_TIPO, COL_SETOR, COL_CAT, COL_SUB, COL_ATIV, COL_ID = 0, 1, 2, 3, 5, 8

EMISSION = "Emissão"
LAND_USE_SECTOR = "Mudança de Uso da Terra e Floresta"
RESIDENTIAL_CATEGORY = "Residencial"
ROAD_SUBCATEGORY = "Rodoviário"
PASSENGER_ACTIVITY = "Transporte de passageiros"

SCOPES = ["gross", "firm_broad", "firm_narrow"]

# Only geographies already verified in this repo; everything else falls back to
# national IPCA unless --ipca-geo is given (SIDRA geography, e.g. n7/3501).
IPCA_GEO_BY_ACP = {"BELO HORIZONTE": "n7/3101"}
NATIONAL_IPCA_GEO = "n1/1"
LEGACY_SLUGS = {"BELO HORIZONTE": "bh"}


def slugify(acp: str) -> str:
    if acp in LEGACY_SLUGS:
        return LEGACY_SLUGS[acp]
    ascii_name = unicodedata.normalize("NFKD", acp).encode("ascii", "ignore").decode()
    return ascii_name.lower().replace(" ", "_")


def _label(v) -> str:
    return "" if v is None else str(v).strip()


# ── SEEG ─────────────────────────────────────────────────────────────────────

def load_acp_municipalities() -> pd.DataFrame:
    df = pd.read_csv(ACP_MUN_CODES_PATH, sep=";", dtype=str)
    df["cod_mun"] = df["cod_mun"].str.strip()
    df["ACPs"] = df["ACPs"].str.strip()
    return df


def build_cache(xlsx_path: Path = XLSX_PATH, cache_path: Path = CACHE_PATH) -> None:
    """Stream the SEEG workbook once and write per-ACP sums (wide, one column per year)."""
    import openpyxl

    acps = load_acp_municipalities()
    id_to_acp = {SEEG_ID_PREFIX + c: a for c, a in zip(acps["cod_mun"], acps["ACPs"])}

    print(f"Reading {xlsx_path.name} (large workbook, this takes a few minutes)...")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    rows = wb["Dados"].iter_rows(values_only=True)
    header = next(rows)
    year_cols = [(i, h) for i, h in enumerate(header) if isinstance(h, int) and h >= CACHE_MIN_YEAR]
    years = [y for _, y in year_cols]

    sums: dict[tuple, np.ndarray] = {}
    for r in rows:
        acp = id_to_acp.get(_label(r[COL_ID]))
        if acp is None:
            continue
        key = (acp, _label(r[COL_TIPO]), _label(r[COL_SETOR]), _label(r[COL_CAT]),
               _label(r[COL_SUB]), _label(r[COL_ATIV]))
        vals = np.array([r[i] or 0.0 for i, _ in year_cols], dtype=float)
        if key in sums:
            sums[key] += vals
        else:
            sums[key] = vals

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(KEY_COLS + years)
        for key, vals in sorted(sums.items()):
            w.writerow(list(key) + [float(v) for v in vals])
    print(f"Saved: {cache_path} ({len(sums)} rows, {years[0]}-{years[-1]})")


def load_seeg(acp: str, rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not CACHE_PATH.exists():
        build_cache()
    df = pd.read_csv(CACHE_PATH, encoding="utf-8", keep_default_na=False)
    if acp not in set(df["acp"]):
        raise ValueError(f"ACP '{acp}' not in SEEG cache. Valid: {sorted(set(df['acp']))}")
    df = df[df["acp"] == acp]
    df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]
    return df


def emissions_by_scope(df: pd.DataFrame) -> pd.DataFrame:
    """tCO2e per year for each scope; rows = year."""
    year_cols = [c for c in df.columns if isinstance(c, int)]
    gross = df[df["tipo"] == EMISSION]
    broad = gross[(gross["cat"] != RESIDENTIAL_CATEGORY) & (gross["setor"] != LAND_USE_SECTOR)]
    road_passenger = (broad["sub"] == ROAD_SUBCATEGORY) & (broad["ativ"] == PASSENGER_ACTIVITY)
    narrow = broad[~road_passenger]
    out = pd.DataFrame({
        "gross": gross[year_cols].sum(),
        "firm_broad": broad[year_cols].sum(),
        "firm_narrow": narrow[year_cols].sum(),
    })
    out.index.name = "year"
    return out


# ── GDP and deflator ─────────────────────────────────────────────────────────

def _read_series(path: Path, col: str) -> pd.Series:
    df = pd.read_csv(path, dtype={"period": str})
    return df.set_index("period")[col]


def _write_series(path: Path, series: pd.Series, col: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["period", col])
        for k, v in series.items():
            w.writerow([k, v])


def load_gdp_nominal(acp: str) -> pd.Series:
    """Nominal GDP (thousand R$) summed over the ACP's municipalities, index = year (int)."""
    path = DATA_DIR / f"{slugify(acp)}_gdp_level.csv"
    if not path.exists():
        codes = load_acp_municipalities().query("ACPs == @acp")["cod_mun"].tolist()
        if not codes:
            raise ValueError(f"No municipalities for ACP '{acp}' in {ACP_MUN_CODES_PATH}")
        print(f"Fetching PIB Municipal for {len(codes)} municipalities of {acp}...")
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for code in codes:
            url = (f"https://apisidra.ibge.gov.br/values/t/{PIB_MUNICIPAL_TABLE}/n6/{code}"
                   f"/v/{PIB_MUNICIPAL_VAR}/p/all")
            for row in _fetch_json(url)[1:]:
                year, val = row["D3C"], row["V"]
                if val not in MISSING_MARKERS:
                    totals[year] = totals.get(year, 0.0) + float(val)
                    counts[year] = counts.get(year, 0) + 1
        partial = sorted(y for y, n in counts.items() if n != len(codes))
        if partial:
            print(f"  Dropping years with missing municipalities (partial sums): {partial}")
        series = pd.Series({y: v for y, v in sorted(totals.items()) if y not in partial})
        _write_series(path, series, "gdp_level")
        print(f"  Saved: {path}")
    gdp = _read_series(path, "gdp_level")
    gdp.index = gdp.index.astype(int)
    return gdp


def load_ipca_monthly(geo: str) -> pd.Series:
    """Monthly IPCA variation (fraction), index = 'YYYYMM', for a SIDRA geography (e.g. n7/3101)."""
    path = DATA_DIR / ("bh_inflation.csv" if geo == "n7/3101" else f"ipca_{geo.replace('/', '_')}.csv")
    if not path.exists():
        print(f"Fetching IPCA for {geo}...")
        series = {}
        for table_id, var_id, start, end in IPCA_TABLES:
            period = str(start) if end is None else f"{start}-{end}"
            url = (f"https://apisidra.ibge.gov.br/values/t/{table_id}/{geo}"
                   f"/v/{var_id}/p/{period}/{IPCA_CLASSIF}")
            for row in _fetch_json(url)[1:]:
                month, val = row["D3C"], row["V"]
                if month >= "201001" and val not in MISSING_MARKERS:
                    series[month] = float(val) / 100
        _write_series(path, pd.Series(dict(sorted(series.items()))), "inflation")
        print(f"  Saved: {path}")
    return _read_series(path, "inflation").sort_index()


def deflator_to_base(ipca_monthly: pd.Series, base_year: int) -> pd.Series:
    """Factor by year that converts nominal values to base_year prices (annual-mean price level)."""
    level = (1 + ipca_monthly).cumprod()
    years = level.index.str[:4].astype(int)
    complete = pd.Series(years).value_counts()
    complete = complete[complete == 12].index
    annual = level.groupby(years).mean()
    annual = annual[annual.index.isin(complete)]
    if base_year not in annual.index:
        raise ValueError(f"base year {base_year} lacks a full 12 months of IPCA "
                         f"(complete years: {annual.index.min()}-{annual.index.max()})")
    return annual[base_year] / annual


# ── TABLE / STATS ────────────────────────────────────────────────────────────

def build_table(acp: str, start: int, end: int, base_year: int, ipca_geo: str | None,
                rebuild_cache: bool = False) -> pd.DataFrame:
    emissions = emissions_by_scope(load_seeg(acp, rebuild_cache))
    gdp_nominal = load_gdp_nominal(acp)
    geo = ipca_geo or IPCA_GEO_BY_ACP.get(acp)
    if geo is None:
        geo = NATIONAL_IPCA_GEO
        print(f"WARNING: no regional IPCA configured for {acp}; using national IPCA "
              f"(pass --ipca-geo, e.g. n7/3501, to override).")
    deflator = deflator_to_base(load_ipca_monthly(geo), base_year)

    years = pd.Index(range(start, end + 1), name="year")
    table = emissions.reindex(years).add_prefix("emissions_")
    gdp_real = (gdp_nominal * deflator).reindex(years) / 1000  # thousand R$ -> R$ million
    table["gdp_real"] = gdp_real
    for scope in SCOPES:
        table[f"intensity_{scope}"] = table[f"emissions_{scope}"] / gdp_real
    return table


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ([f"emissions_{s}" for s in SCOPES] + ["gdp_real"] + [f"intensity_{s}" for s in SCOPES])
    for m in metrics:
        s = table[m].dropna()
        rows.append({"metric": m, "mean": s.mean(), "std": s.std(),
                     "n_years": len(s), "first_year": s.index.min(), "last_year": s.index.max()})
    return pd.DataFrame(rows)


# ── PLOT ─────────────────────────────────────────────────────────────────────

EMP_COLOR = "#d73027"
GROSS_COLOR = "#555555"


def plot(table: pd.DataFrame, acp: str, base_year: int, out_stem: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    panels = [
        ("emissions_", 1e6, "Aggregate emissions (MtCO$_2$e)"),
        ("intensity_", 1.0, f"Emissions / real GDP\n(tCO$_2$e per R\\$ million, {base_year} prices)"),
    ]
    for ax, (prefix, scale, ylabel) in zip(axes, panels):
        lo, hi = table[prefix + "firm_narrow"] / scale, table[prefix + "firm_broad"] / scale
        gross = table[prefix + "gross"] / scale
        ax.fill_between(table.index, lo, hi, color=EMP_COLOR, alpha=0.25, lw=0)
        ax.plot(table.index, lo, color=EMP_COLOR, lw=1.2)
        ax.plot(table.index, hi, color=EMP_COLOR, lw=1.2)
        ax.plot(table.index, gross, color=GROSS_COLOR, lw=1.2, ls="--")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_xlim(table.index.min(), gross.dropna().index.max())
        ax.grid(axis="y", lw=0.4, alpha=0.4)

    band = plt.Rectangle((0, 0), 1, 1, fc=EMP_COLOR, alpha=0.25, ec=EMP_COLOR)
    gross_line = plt.Line2D([], [], color=GROSS_COLOR, lw=1.2, ls="--")
    fig.legend([band, gross_line],
               ["Firm scope (excl. household; lower edge also excl. road passenger transport)",
                "Gross total (all SEEG emissions)"],
               loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{acp.title()} metro region - SEEG emissions (CO$_2$e, GWP-100)", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_stem}.pdf / .png")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acp", default="BELO HORIZONTE", help="ACP name as in ACPs_MUN_CODES.csv")
    p.add_argument("--start", type=int, default=2010)
    p.add_argument("--end", type=int, default=2024)
    p.add_argument("--base-year", type=int, default=2024, help="Price base year for real GDP")
    p.add_argument("--ipca-geo", default=None, help="SIDRA geography for IPCA, e.g. n7/3101, n6/5300108, n1/1")
    p.add_argument("--rebuild-cache", action="store_true", help="Re-read the SEEG workbook")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    acp = args.acp.upper()
    table = build_table(acp, args.start, args.end, args.base_year, args.ipca_geo, args.rebuild_cache)
    stats = summarize(table)

    slug = slugify(acp)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"seeg_emissions_series_{slug}.csv")
    stats.to_csv(RESULTS_DIR / f"seeg_emissions_summary_{slug}.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.2f}".format)
    print(f"\n{acp}, {args.start}-{args.end}, real GDP in {args.base_year} prices "
          f"(std = sample std, ddof=1)")
    print(table.to_string())
    print()
    print(stats.to_string(index=False))

    if not args.no_plot:
        plot(table, acp, args.base_year, RESULTS_DIR / f"seeg_emissions_{slug}")


if __name__ == "__main__":
    main()
