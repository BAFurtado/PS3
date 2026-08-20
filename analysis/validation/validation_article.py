"""
Validation charts for the Cambridge JRES article.

Figures produced (saved to analysis/validation/results/):
    fig1_timeseries.pdf   — Normalised time-series: unemployment, inflation, GDP growth
    fig2_stylized.pdf     — Stylized facts: Okun's Law and Phillips Curve
    fig3_sector_comp.pdf  — Sector employment composition by ACP (RAIS 2010)

Run from project root:
    python -m analysis.validation.validation_article
"""

import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# ── PATHS ─────────────────────────────────────────────────────────────────────

# The 27-city batch (26 cities x 6 configs x 10 seeds, plus Sao Paulo x 6 configs x
# 6 seeds = 1,596 sims), the same file text_density's prepare_data.py reads.
SIM_PATH  = os.path.join(ROOT, "output", "final_stats_27cities.csv")
EMP_PATH  = os.path.join(ROOT, "analysis", "validation", "real_world_data", "real_data_macroeconomic.csv")
RAIS_PATH = os.path.join(ROOT, "analysis", "validation", "real_world_data", "mun_isic12_2010.csv")
ACP_PATH  = os.path.join(ROOT, "input", "ACPs_MUN_CODES.csv")
OUT_DIR   = os.path.join(ROOT, "analysis", "validation", "results")

VALIDATION_END = "2024-12-01"
BURN_IN_MONTHS = 24   # skip first N months of each run to avoid initialisation artefacts

SIMULATION_ACPS = [
    'ARACAJU', 'BELEM', 'BELO HORIZONTE', 'BOA VISTA', 'BRASILIA',
    'CAMPO GRANDE', 'CUIABA', 'CURITIBA', 'FLORIANOPOLIS', 'FORTALEZA',
    'GOIANIA', 'JOAO PESSOA', 'MACAPA', 'MACEIO', 'MANAUS', 'NATAL',
    'PALMAS', 'PORTO ALEGRE', 'PORTO VELHO', 'RECIFE', 'RIO BRANCO',
    'RIO DE JANEIRO', 'SALVADOR', 'SAO LUIS', 'SAO PAULO', 'TERESINA',
    'VITORIA',
]

REGION_ORDER = {
    'Norte':        ['BELEM', 'BOA VISTA', 'MACAPA', 'MANAUS', 'PALMAS', 'PORTO VELHO', 'RIO BRANCO'],
    'Nordeste':     ['ARACAJU', 'FORTALEZA', 'JOAO PESSOA', 'MACEIO', 'NATAL', 'RECIFE', 'SALVADOR', 'SAO LUIS', 'TERESINA'],
    'Centro-Oeste': ['BRASILIA', 'CAMPO GRANDE', 'CUIABA', 'GOIANIA'],
    'Sudeste':      ['BELO HORIZONTE', 'RIO DE JANEIRO', 'SAO PAULO', 'VITORIA'],
    'Sul':          ['CURITIBA', 'FLORIANOPOLIS', 'PORTO ALEGRE'],
}

REGION_COLORS = {
    'Norte':        '#d7191c',
    'Nordeste':     '#fdae61',
    'Centro-Oeste': '#1a9641',
    'Sudeste':      '#2c7bb6',
    'Sul':          '#abd9e9',
}

SECTOR_DISPLAY = {
    'Agriculture':   'Agriculture',
    'Business':      'Business Svcs',
    'Construction':  'Construction',
    'Financial':     'Financial',
    'Government':    'Government',
    'Manufacturing': 'Manufacturing',
    'Mining':        'Mining',
    'OtherServices': 'Other Services',
    'RealEstate':    'Real Estate',
    'Trade':         'Trade',
    'Transport':     'Transport',
    'Utilities':     'Utilities',
}

# ── STYLE ─────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "font.family":     "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SIM_COLOR = "#2166ac"
EMP_COLOR = "#d73027"

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_simulation(path=SIM_PATH, burn_in=BURN_IN_MONTHS, end=VALIDATION_END):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["simulation_id", "month"]).reset_index(drop=True)
    rank = df.groupby("simulation_id").cumcount()
    df = df[rank >= burn_in].reset_index(drop=True)
    df = df[df["month"] <= pd.Timestamp(end)].copy()
    return df


def load_empirical(path=EMP_PATH):
    raw = pd.read_csv(path, sep=";", encoding="latin-1")
    raw.columns = ["month", "inflation", "gdp_growth", "income_growth", "unemployment"]
    # Month column arrives as float "2010.01" → parse year and month parts
    raw["month"] = raw["month"].apply(
        lambda v: pd.Timestamp(year=int(v), month=round((v % 1) * 100), day=1)
        if pd.notna(v) else pd.NaT
    )
    for col in ["inflation", "gdp_growth", "income_growth", "unemployment"]:
        raw[col] = pd.to_numeric(
            raw[col].astype(str).str.replace(",", "."), errors="coerce"
        )
    # Both inflation and unemployment are in % — convert to fractions
    raw["inflation"]    /= 100
    raw["unemployment"] /= 100
    # gdp_growth is already a monthly rate in decimal form, but it is the month-on-month
    # growth of monthly GDP at CURRENT prices: compounded it is positive in every year
    # 2011-2024, including 2015, 2016 and 2020 when real GDP contracted, and it carries an
    # unadjusted calendar profile (Jan -5.5%, Mar +6.6% on 15-year averages). The model runs
    # in real terms with near-zero inflation, so comparing the two directly overstates the
    # model's fit: nominal averages 10.0%/yr against a simulated 10.2%/yr, while real
    # activity averages 3.9%/yr. Deflate by the IPCA column of the same file to put both
    # sides in real terms.
    raw["gdp_growth"] = (1 + raw["gdp_growth"]) / (1 + raw["inflation"]) - 1
    return raw.dropna(subset=["month"]).reset_index(drop=True)


def load_rais_sector(rais_path=RAIS_PATH, acp_path=ACP_PATH, acps=SIMULATION_ACPS):
    rais = pd.read_csv(rais_path)
    acp  = pd.read_csv(acp_path, sep=";")
    acp["cod_mun6"] = acp["cod_mun"] // 10
    merged = rais.merge(acp, left_on="codemun", right_on="cod_mun6", how="inner")
    merged = merged[merged["ACPs"].isin(acps)]
    agg = (
        merged.groupby(["ACPs", "isic_r4"])["qtde_vinc_ativos_sum"]
              .sum()
              .reset_index()
    )
    totals = agg.groupby("ACPs")["qtde_vinc_ativos_sum"].sum().rename("total")
    agg = agg.join(totals, on="ACPs")
    agg["share"] = agg["qtde_vinc_ativos_sum"] / agg["total"]
    return agg


IVG_R_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.21340/dados?formato=csv"
)
IVG_R_CACHE = os.path.join(OUT_DIR, "ivgr_cache.csv")


def load_ivgr(url=IVG_R_URL, cache_path=IVG_R_CACHE):
    """
    Download BCB IVG-R (residential real estate price index, base Mar/2001=100).
    Caches locally so subsequent runs don't need a network call.
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        print(f"  IVG-R loaded from cache ({cache_path})")
        return df

    print(f"  Downloading IVG-R from BCB API ...")
    raw = pd.read_csv(url, sep=";", encoding="latin-1")
    raw.columns = ["date", "value"]
    raw["date"]  = pd.to_datetime(raw["date"], format="%d/%m/%Y")
    raw["value"] = pd.to_numeric(
        raw["value"].astype(str).str.replace(",", "."), errors="coerce"
    )
    df = raw.dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  IVG-R cached to {cache_path}")
    return df


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _sim_band(df, col):
    """Monthly median and 5th/95th percentiles across all runs."""
    g = df.groupby("month")[col]
    return g.median(), g.quantile(0.05), g.quantile(0.95)


def _zscore_series(s, ref=None):
    """Z-score s using ref's mean/std (defaults to s itself)."""
    src = ref if ref is not None else s
    mu, sigma = src.mean(), src.std()
    if sigma == 0:
        return s - mu
    return (s - mu) / sigma


def _city_order():
    order = []
    for cities in REGION_ORDER.values():
        order.extend(cities)
    return order


def _city_region_map():
    m = {}
    for region, cities in REGION_ORDER.items():
        for c in cities:
            m[c] = region
    return m


def _ols_line(ax, x, y, color, lw=1.5, clip_pct=(0.02, 0.98)):
    """Fit OLS on finite values, draw line clipped to given percentile range."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_, y_ = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x_) < 10:
        return
    slope, intercept, r, p, _ = stats.linregress(x_, y_)
    lo = np.quantile(x_, clip_pct[0])
    hi = np.quantile(x_, clip_pct[1])
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, slope * xs + intercept, color=color, lw=lw)
    ax.annotate(
        f"$\\hat{{\\beta}}$={slope:.3f}",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8,
        color=color,
    )


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {os.path.join(OUT_DIR, name)}.pdf / .png")


# ── FIGURE 1: TIME-SERIES VALIDATION ─────────────────────────────────────────

TS_VARS = [
    dict(sim="inflation",       emp="inflation",   label="Monthly inflation"),
    dict(sim="gdp_growth_rate", emp="gdp_growth",  label="GDP growth rate"),
]


def plot_timeseries(sim_df, emp_df, normalize=True):
    """
    2-panel time-series comparison (inflation + GDP growth).
    Unemployment omitted due to structural level mismatch.

    normalize=True  — each series z-scored to its own mean/std; CI bands share
                      the simulated median's parameters so their shape is preserved.
                      Focuses the comparison on dynamics rather than levels.
    normalize=False — raw units; simulated on the left axis, empirical on an
                      independent right axis so scale differences don't obscure
                      co-movement. Level mismatch is explicit and honest.
    """
    suffix = "fig1_timeseries_normalized" if normalize else "fig1_timeseries_raw"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)

    sim_df = sim_df.copy()
    lo, hi = sim_df["gdp_growth_rate"].quantile(0.01), sim_df["gdp_growth_rate"].quantile(0.99)
    sim_df["gdp_growth_rate"] = sim_df["gdp_growth_rate"].clip(lo, hi)

    for ax, cfg in zip(axes, TS_VARS):
        med, p05, p95 = _sim_band(sim_df, cfg["sim"])
        emp_s = emp_df.set_index("month")[cfg["emp"]].dropna()

        common = med.index.intersection(emp_s.index)
        med_a, p05_a, p95_a = med[common], p05[common], p95[common]
        emp_a = emp_s[common]

        if normalize:
            mu_sim, sd_sim = med_a.mean(), med_a.std()
            sd_sim = sd_sim if sd_sim > 0 else 1.0
            med_plot = (med_a - mu_sim) / sd_sim
            p05_plot = (p05_a - mu_sim) / sd_sim
            p95_plot = (p95_a - mu_sim) / sd_sim
            emp_plot = _zscore_series(emp_a)

            ax.fill_between(common, p05_plot, p95_plot, color=SIM_COLOR, alpha=0.2)
            ax.plot(common, med_plot, color=SIM_COLOR, lw=1.5)
            ax.plot(common, emp_plot, color=EMP_COLOR, lw=1.5, ls="--")
            ax.axhline(0, color="grey", lw=0.4, ls=":")
            ax.set_ylabel("z-score", fontsize=8)
        else:
            # Raw: simulated on left axis, empirical on independent right axis
            ax.fill_between(common, p05_a, p95_a, color=SIM_COLOR, alpha=0.2)
            ax.plot(common, med_a, color=SIM_COLOR, lw=1.5)
            ax.set_ylabel(cfg["label"], color=SIM_COLOR, fontsize=8)
            ax.tick_params(axis="y", colors=SIM_COLOR, labelsize=7)

            ax2 = ax.twinx()
            ax2.plot(common, emp_a, color=EMP_COLOR, lw=1.5, ls="--")
            ax2.set_ylabel("Empirical", color=EMP_COLOR, fontsize=7)
            ax2.tick_params(axis="y", colors=EMP_COLOR, labelsize=7)
            ax2.spines["right"].set_visible(True)

        ax.set_title(cfg["label"], fontsize=10)
        ax.tick_params(axis="x", rotation=40, labelsize=8)

    sim_patch = mpatches.Patch(color=SIM_COLOR, alpha=0.6, label="Simulated (median + 90% CI)")
    emp_patch  = mpatches.Patch(color=EMP_COLOR, label="Empirical")
    fig.legend(handles=[sim_patch, emp_patch], loc="lower center",
               ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    _save(fig, suffix)


# ── FIGURE 2: STYLIZED FACTS ──────────────────────────────────────────────────

def plot_stylized_facts(sim_df):
    """
    Two panels showing simulated macroeconomic relationships:
      Left  — Okun's Law: Δunemployment vs. GDP growth rate
      Right — Phillips Curve: monthly inflation vs. unemployment rate

    Empirical overlays are omitted because the model's unemployment level
    (calibrated to internal dynamics) sits in a different range than national
    PNAD estimates; overlaying them would conflate a level-calibration issue
    with the structural relationship being validated. The claim is that the
    model internally exhibits the correct sign and direction, not that levels match.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    sim = sim_df.copy().sort_values(["simulation_id", "month"])
    sim["delta_unemp"] = sim.groupby("simulation_id")["unemployment"].diff()
    sim = sim.dropna(subset=["delta_unemp", "gdp_growth_rate", "inflation"])

    for col, lo, hi in [
        ("gdp_growth_rate", 0.01, 0.99),
        ("delta_unemp",     0.01, 0.99),
        ("unemployment",    0.01, 0.99),
        ("inflation",       0.01, 0.99),
    ]:
        q_lo, q_hi = sim[col].quantile(lo), sim[col].quantile(hi)
        sim[col] = sim[col].clip(q_lo, q_hi)

    # ── OKUN'S LAW ──────────────────────────────────────────────────────────
    ax = axes[0]
    hb = ax.hexbin(sim["gdp_growth_rate"], sim["delta_unemp"],
                   gridsize=35, cmap="Blues", mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax, label="Count", shrink=0.6, pad=0.02, aspect=20)
    cb.ax.tick_params(labelsize=7)

    _ols_line(ax, sim["gdp_growth_rate"], sim["delta_unemp"], EMP_COLOR, lw=2.0)

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("GDP growth rate (monthly)", fontsize=9)
    ax.set_ylabel(r"$\Delta$ Unemployment rate", fontsize=9)
    ax.set_title("Okun's Law", fontsize=10)
    ax.grid(False)
    ax.set_facecolor("white")

    # ── PHILLIPS CURVE ───────────────────────────────────────────────────────
    ax = axes[1]
    hb2 = ax.hexbin(sim["unemployment"], sim["inflation"],
                    gridsize=35, cmap="Blues", mincnt=1, linewidths=0.2)
    cb2 = fig.colorbar(hb2, ax=ax, label="Count", shrink=0.6, pad=0.02, aspect=20)
    cb2.ax.tick_params(labelsize=7)

    _ols_line(ax, sim["unemployment"], sim["inflation"], EMP_COLOR, lw=2.0)

    ax.set_xlabel("Unemployment rate", fontsize=9)
    ax.set_ylabel("Monthly inflation rate", fontsize=9)
    ax.set_title("Phillips Curve (short-run)", fontsize=10)
    ax.grid(False)
    ax.set_facecolor("white")

    plt.tight_layout()
    _save(fig, "fig2_stylized")


# ── FIGURE 3: SECTOR COMPOSITION ──────────────────────────────────────────────

def plot_sector_composition(sector_df):
    """
    Bubble matrix: cities on x-axis (ordered Norte→Sul) × sectors on y-axis.
    Bubble size proportional to employment share. Colour by Brazilian region.
    """
    city_order   = _city_order()
    city_region  = _city_region_map()
    sector_order = sorted(SECTOR_DISPLAY.keys())

    # Filter to simulation ACPs only and enforce ordering
    df = sector_df[sector_df["ACPs"].isin(city_order)].copy()

    n_cities  = len(city_order)
    n_sectors = len(sector_order)

    fig, ax = plt.subplots(figsize=(17, 5))

    max_share = df["share"].max()
    MAX_AREA  = 900   # max bubble area in points²

    for _, row in df.iterrows():
        city   = row["ACPs"]
        sector = row["isic_r4"]
        if city not in city_order or sector not in sector_order:
            continue
        xi = city_order.index(city)
        yi = sector_order.index(sector)
        area   = (row["share"] / max_share) * MAX_AREA
        region = city_region.get(city, "Norte")
        ax.scatter(xi, yi,
                   s=area,
                   color=REGION_COLORS[region],
                   alpha=0.80,
                   edgecolors="grey",
                   linewidths=0.4,
                   zorder=3)

    # Axes
    ax.set_xticks(range(n_cities))
    ax.set_xticklabels(
        [c.title() for c in city_order],
        rotation=55, ha="right", fontsize=7.5,
    )
    ax.set_yticks(range(n_sectors))
    ax.set_yticklabels(
        [SECTOR_DISPLAY.get(s, s) for s in sector_order],
        fontsize=9,
    )
    ax.set_xlim(-0.8, n_cities - 0.2)
    ax.set_ylim(-0.8, n_sectors - 0.2)
    ax.grid(True, lw=0.3, alpha=0.35, zorder=0)

    # Region legend (colour)
    region_patches = [
        mpatches.Patch(color=c, label=r)
        for r, c in REGION_COLORS.items()
    ]

    # Size reference legend
    ref_shares = [0.05, 0.15, 0.30]
    size_handles = [
        plt.scatter([], [],
                    s=(s / max_share) * MAX_AREA,
                    color="grey", alpha=0.6,
                    label=f"{s:.0%} share")
        for s in ref_shares
    ]

    leg1 = ax.legend(handles=region_patches, title="Region",
                     fontsize=8, loc="lower right", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, title="Emp. share",
              fontsize=8, loc="upper right", framealpha=0.9)

    ax.set_title(
        "Sectoral employment composition across metropolitan areas — RAIS 2010\n"
        "(bubble size = sector share of ACP formal employment)",
        fontsize=10,
    )

    # Region dividers
    cumulative = 0
    for region, cities in REGION_ORDER.items():
        n = len(cities)
        if cumulative > 0:
            ax.axvline(cumulative - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.text(
            cumulative + n / 2 - 0.5, n_sectors - 0.1,
            region, ha="center", va="bottom", fontsize=7, style="italic", alpha=0.7,
        )
        cumulative += n

    plt.tight_layout()
    _save(fig, "fig3_sector_composition")


# ── FIGURE 4: HOUSE PRICE DYNAMICS vs IVG-R ──────────────────────────────────

BASE_YEAR = 2012   # rebase both series to this year's average = 100 (matches 24-month burn-in)


def _rebase(series: pd.Series, base_year: int) -> pd.Series:
    """Reindex a time-indexed series so that the mean over base_year = 100."""
    mask = series.index.year == base_year
    base = series[mask].mean()
    if base == 0 or np.isnan(base):
        return series
    return series / base * 100


def plot_house_prices(sim_df, ivgr_df):
    """
    Two-panel figure:
      Left  — Simulated house price index (median + 90% CI, rebased to 2011=100)
               vs BCB IVG-R (rebased to 2011=100).
      Right — Simulated rent-to-price ratio over time (no empirical equivalent;
               shown as a housing market stylized fact).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    sim = sim_df.copy()
    sim["month"] = pd.to_datetime(sim["month"])

    # ── LEFT: price index comparison ─────────────────────────────────────────
    ax = axes[0]

    # Simulated: median house_price per month across all cities and runs
    med, p05, p95 = _sim_band(sim, "house_price")
    med  = _rebase(med,  BASE_YEAR)
    p05  = _rebase(p05,  BASE_YEAR)
    p95  = _rebase(p95,  BASE_YEAR)

    ax.fill_between(med.index, p05, p95, color=SIM_COLOR, alpha=0.20)
    ax.plot(med.index, med, color=SIM_COLOR, lw=1.8, label="Simulated (median + 90% CI)")

    # IVG-R: clip to validation window, rebase
    ivgr = ivgr_df.set_index("date")["value"]
    ivgr = ivgr[(ivgr.index >= sim["month"].min()) &
                (ivgr.index <= pd.Timestamp(VALIDATION_END))]
    ivgr_r = _rebase(ivgr, BASE_YEAR)

    ax.plot(ivgr_r.index, ivgr_r, color=EMP_COLOR, lw=1.8, ls="--",
            label="IVG-R — BCB (empirical)")

    ax.axhline(100, color="grey", lw=0.4, ls=":")
    ax.set_ylabel(f"Price index ({BASE_YEAR} avg = 100)", fontsize=9)
    ax.set_title("House price dynamics", fontsize=10)
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.legend(fontsize=8)

    # ── RIGHT: rent-to-price ratio (simulated) ───────────────────────────────
    ax = axes[1]

    # Compute rent/price ratio per run-month, then band
    sim["rent_price_ratio"] = sim["house_rent"] / sim["house_price"]
    rp_med, rp_p05, rp_p95 = _sim_band(sim, "rent_price_ratio")

    ax.fill_between(rp_med.index, rp_p05, rp_p95, color=SIM_COLOR, alpha=0.20)
    ax.plot(rp_med.index, rp_med, color=SIM_COLOR, lw=1.8)

    ax.set_ylabel("Rent / Price ratio (monthly)", fontsize=9)
    ax.set_title("Simulated rent-to-price ratio", fontsize=10)
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.text(0.97, 0.95, "Simulated only\n(no Brazilian rental\nyield time-series available)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="grey", style="italic")

    fig.suptitle(
        "Figure 4 — Housing market dynamics: simulated vs. empirical (IVG-R/BCB)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    _save(fig, "fig4_house_prices")


# ── TABLE: HOUSING VALIDATION ─────────────────────────────────────────────────

# Empirical benchmarks: (label, emp_lo, emp_hi, source_tag, sim_col, fmt)
# source_tag maps to footnotes defined in FOOTNOTES below.
HOUSING_TABLE_ROWS = [
    ("Housing stock / GDP",
     1.5,  2.5,  "a", "housing_stock_gdp",           ".2f"),
    ("Price-to-monthly-wage ratio",
     60,   150,  "b", "price_wage",                  ".0f"),
    ("Price-to-annual-income ratio",
     6.0,  12.0, "c", "price_income",                ".1f"),
    ("Vacancy rate",
     0.08, 0.13, "d", "vacancy",                     ".2f"),
    ("Consumption / GDP",
     0.55, 0.65, "e", "consumption_gdp",             ".2f"),
    ("Housing production (per 1,000 hab.)",
     2.0,  4.0,  "f", "housing_production_per_1000", ".1f"),
    ("Housing stock / permanent income",
     5.0,  8.0,  "g", "housing_stock_perm_income",   ".1f"),
]

FOOTNOTES = {
    "a": "Banco Central do Brasil, Relat\\'orio de Estabilidade Financeira (2019).",
    "b": "Funda\\c{c}\\~ao Jo\\~ao Pinheiro, D\\'eficit Habitacional no Brasil (2022).",
    "c": "Banco Central do Brasil, Relat\\'orio de Estabilidade Financeira (2023).",
    "d": "IBGE, Censo Demogr\\'afico 2010 and 2022.",
    "e": "IBGE, Contas Nacionais (2010--2024 average).",
    "f": "C\\^amara Brasileira da Ind\\'ustria da Constru\\c{c}\\~ao -- CBIC (2019).",
    "g": "Authors' estimate based on national household survey data.",
}


def _compute_housing_moments(sim_df):
    """
    Compute per-run time-means of each housing validation indicator.
    Returns a DataFrame indexed by simulation_id.
    """
    df = sim_df.copy().sort_values(["simulation_id", "month"])

    df["housing_stock_value"]     = df["number_domiciles"] * df["house_price"]
    df["housing_stock_gdp"]       = df["housing_stock_value"] / (df["gdp_level"] * 12)
    df["price_wage"]              = df["house_price"] / df["firms_wage_per_worker"]
    df["price_income"]            = df["house_price"] / (df["families_wages_received"] * 12)
    df["vacancy"]                 = df["house_vacancy"]
    # Households, not the dwelling stock: average_utility averages over non-empty
    # families and vacant houses hold none. See housing_validation.py for the full note.
    df["consumption_gdp"]         = (
        df["average_utility"] * df["number_domiciles"] * (1 - df["house_vacancy"])
    ) / df["gdp_level"]
    df["new_houses"] = (
        df.groupby("simulation_id")["number_domiciles"]
          .diff()
          .clip(lower=0)
    )
    df["housing_production_per_1000"] = df["new_houses"] * 12 * 1000 / df["pop"]
    # Denominator is aggregate family income, so it scales with households, not dwellings.
    df["housing_stock_perm_income"]   = df["housing_stock_value"] / (
        df["families_median_permanent_income"] * 12
        * df["number_domiciles"] * (1 - df["house_vacancy"])
    )

    sim_cols = [r[4] for r in HOUSING_TABLE_ROWS]
    for c in sim_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df.groupby("simulation_id")[sim_cols].mean()


def write_housing_latex(sim_df, caption=None, label="tab:housing_validation"):
    """
    Write housing validation LaTeX table to results/housing_validation.tex.

    Each row shows: indicator | empirical range | simulated (median, p10--p90) | in range? | source
    Rows where the simulated p10--p90 falls entirely outside the empirical range
    are flagged with a dagger (\\dag).
    """
    moments = _compute_housing_moments(sim_df)

    if caption is None:
        caption = (
            "Housing market validation: simulated moments compared to "
            "empirical benchmarks. Simulated values report the median and "
            "the 10th--90th percentile range across all simulation runs. "
            "\\dag~indicates that the simulated range falls outside the "
            "empirical benchmark."
        )

    rows = []
    for label_str, emp_lo, emp_hi, src, col, fmt in HOUSING_TABLE_ROWS:
        col_data = moments[col].dropna()
        med  = col_data.median()
        p10  = col_data.quantile(0.10)
        p90  = col_data.quantile(0.90)

        # Flag if simulated p10-p90 band does not overlap empirical range at all
        out_of_range = (p90 < emp_lo) or (p10 > emp_hi)
        flag = r"\dag" if out_of_range else ""

        # Format empirical range — integers if fmt ends with 'f' and values >= 10
        if emp_lo >= 10:
            emp_str = f"{emp_lo:.0f}--{emp_hi:.0f}"
        else:
            emp_str = f"{emp_lo}--{emp_hi}"

        med_str = format(med, fmt.lstrip(".").replace("f", "") and fmt or fmt)
        rng_str = (
            f"{format(p10, fmt.lstrip('').replace('f','') and fmt or fmt)}"
            f"--{format(p90, fmt.lstrip('').replace('f','') and fmt or fmt)}"
        )

        rows.append(
            f"    {label_str} {flag} & {emp_str} & "
            f"{med_str} ({rng_str}) & \\textsuperscript{{{src}}} \\\\"
        )

    footnote_lines = "\n".join(
        f"    \\textsuperscript{{{k}}}~{v}"
        for k, v in FOOTNOTES.items()
    )

    table = r"""\begin{table}[htbp]
\centering
\small
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\begin{tabular}{p{6.2cm} c c c}
\hline\hline
\textbf{Indicator} & \textbf{Empirical range} & \textbf{Simulated median (p10--p90)} & \textbf{Source} \\
\hline
""" + "\n".join(rows) + r"""
\hline\hline
\multicolumn{4}{p{14cm}}{\footnotesize \textit{Notes:}
""" + footnote_lines + r"""}
\end{tabular}
\end{table}
"""

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "housing_validation.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)

    print(f"  Saved -> {out_path}")
    print()
    print(table)
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading simulation data …")
    sim_df = load_simulation()
    n_runs = sim_df["simulation_id"].nunique()
    print(f"  {len(sim_df):,} rows | {n_runs} runs | "
          f"{sim_df['month'].min().date()} – {sim_df['month'].max().date()}")

    print("Loading empirical macro data …")
    emp_df = load_empirical()
    print(f"  {len(emp_df)} months | "
          f"{emp_df['month'].min().date()} – {emp_df['month'].max().date()}")

    print("Loading RAIS sector data …")
    sector_df = load_rais_sector()
    print(f"  {sector_df['ACPs'].nunique()} ACPs | "
          f"{sector_df['isic_r4'].nunique()} sectors")

    print("Loading IVG-R (BCB house price index) …")
    ivgr_df = load_ivgr()
    print(f"  {len(ivgr_df)} months | "
          f"{ivgr_df['date'].min().date()} - {ivgr_df['date'].max().date()}")

    print("\nGenerating Figure 1a — time series (normalised) ...")
    plot_timeseries(sim_df, emp_df, normalize=True)

    print("Generating Figure 1b — time series (raw, dual axes) ...")
    plot_timeseries(sim_df, emp_df, normalize=False)

    print("Generating Figure 2 — stylized facts (simulated only) ...")
    plot_stylized_facts(sim_df)

    print("Generating Figure 3 — sector composition ...")
    plot_sector_composition(sector_df)

    print("Generating Figure 4 — house prices vs IVG-R ...")
    plot_house_prices(sim_df, ivgr_df)

    print("Generating housing validation LaTeX table ...")
    write_housing_latex(sim_df)

    print("\nDone. All figures in:", OUT_DIR)
