"""Regenerate the paper's data tables on the corrected 26-city, n=10 batch.

Writes papers/density_housing_inequality/tables/{Table3,Table4,Table7,Table8}*.tex.
Tables 1 and 6 (Phillips/Okun) are NOT regenerated: the script behind them is
not in the repository.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from analysis.validation import compare_density_tier2 as C  # noqa: E402
from density_text_numbers import build_panels_window  # noqa: E402

TABLES = ROOT / "papers" / "density_housing_inequality" / "tables"
warnings.filterwarnings("ignore")

EVAL_START, EVAL_END = pd.Timestamp("2011-01-01"), pd.Timestamp("2021-12-01")


def st(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---------------------------------------------------------------------------
print("loading ...")
stats = C.load_stats(C.NEW_STATS, "new")
reg = C.load_regional(C.NEW_REG, "new")

acp, cap, per = build_panels_window(stats, reg,
                                    pd.Timestamp("2035-01-01"),
                                    pd.Timestamp("2039-12-01"))

# ============================================================ Table 3
SPECS = {
    "ACP": (acp, "delta_gini", "delta_house_price", "delta_affordability_median"),
    "Capital": (cap, "delta_regional_gini", "delta_regional_house_values",
                "delta_median_affordability"),
    "Periphery": (per, "delta_gini_periphery", "delta_house_values_periphery",
                  "delta_median_affordability_periphery"),
}
prim, sec = {}, {}
for lvl, (df, y, mp, ms) in SPECS.items():
    prim[lvl] = C.eq3(df, y, mp)
    sec[lvl] = C.eq3(df, y, ms)
    print(lvl, "primary", prim[lvl], "\n", lvl, "secondary", sec[lvl])

L = ["ACP", "Capital", "Periphery"]
# Panel A is now a signed, significant coefficient of order 1e-4: printed x10^4
pa_b = " & ".join(f"{prim[l]['beta'] * 1e4:+.3f}{st(prim[l]['p'])}" for l in L)
pa_se = " & ".join(f"({prim[l]['se'] * 1e4:.3f})" for l in L)
pb_b = " & ".join(f"{sec[l]['beta']:+.4f}{st(sec[l]['p'])}" for l in L)
pb_se = " & ".join(f"({sec[l]['se']:.4f})" for l in L)
r2a = " & ".join(f"{prim[l]['r2']:.3f}" for l in L)
r2b = " & ".join(f"{sec[l]['r2']:.3f}" for l in L)
nn = " & ".join(f"{prim[l]['n']}" for l in L)
kk = " & ".join(f"{prim[l]['n_cities']}" for l in L)

t3 = rf"""\begin{{table}}[htbp]
\centering
\caption{{Housing wealth and affordability channels: regression results (Equation~3)}}
\label{{tab:regression}}
\begin{{threeparttable}}
\begin{{tabular}}{{lccc}}
\toprule
 & ACP & Capital & Periphery \\
\midrule
\multicolumn{{4}}{{l}}{{\textit{{Panel A: Primary mechanism}} ($\Delta M$ = house price change)}} \\[2pt]
$\hat{{\beta}} \times 10^{{4}}$ & {pa_b} \\
 & {pa_se} \\[4pt]
$R^2$ & {r2a} \\
$N$ & {nn} \\
Cities & {kk} \\
\midrule
\multicolumn{{4}}{{l}}{{\textit{{Panel B: Secondary check}} ($\Delta M$ = median rent/income ratio, renters)}} \\[2pt]
$\hat{{\beta}}$ & {pb_b} \\
 & {pb_se} \\[4pt]
$R^2$ & {r2b} \\
$N$ & {nn} \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\footnotesize
\item Notes: OLS with metropolitan-area fixed effects (HC3 standard errors in parentheses).
\item $\Delta M$ and $\Delta Y$ are 60-month window averages (2035--2039) of policy-induced changes (Eq.~1--2).
\item Panel A coefficients and standard errors are multiplied by $10^{{4}}$ for legibility; $\Delta M$ there is a house price in model units, so the coefficient is small by construction.
\item Periphery excludes five single-municipality ACPs (Boa Vista, Campo Grande, Manaus, Palmas, Rio Branco).
\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}"""
(TABLES / "Table3_regression.tex").write_text(t3)
print("wrote Table3")

# ============================================================ Table 4
WINDOWS = {
    "Tail (2035--2039, headline)": ("2035-01-01", "2039-12-01"),
    "Alternative (2027--2031)": ("2027-01-01", "2031-12-01"),
    "Alternative (2028--2030, narrow)": ("2028-01-01", "2030-12-01"),
}
rows4 = []
for label, (a, b) in WINDOWS.items():
    aw, _, _ = build_panels_window(stats, reg, pd.Timestamp(a), pd.Timestamp(b))
    r = C.eq3(aw, "delta_gini", "delta_affordability_median")
    rows4.append((label, aw["delta_gini"].mean(), r))
    print(label, r)

body4 = "\n".join(
    f"{lab} & {m:+.4f} & {r['beta']:+.4f}{st(r['p'])} ({r['se']:.4f}) "
    f"& {r['r2']:.3f} & {r['n']} \\\\"
    for lab, m, r in rows4)

t4 = rf"""\begin{{table}}[htbp]
\centering
\caption{{Affordability-channel coefficient across averaging windows}}
\label{{tab:window_robustness}}
\begin{{threeparttable}}
\begin{{tabular}}{{lcccc}}
\toprule
Window & mean $\Delta Y$ & $\hat{{\beta}}$ & $R^2$ & $N$ \\
\midrule
{body4}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\footnotesize
\item Notes: Equation~(3) (ACP level, secondary spec), re-estimated with the averaging window moved away from the long-run tail to the two earlier windows examined in Figure~\ref{{fig:timeseries}}. HC3 SE in parentheses.
\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}"""
(TABLES / "Table4_window_robustness.tex").write_text(t4)
print("wrote Table4")

# ---------------------------------------------------------------------------
# The stationarity table (ADF/KPSS on the pooled monthly mean dY) was dropped
# from the paper: it was a lightweight complement to the window check, nothing
# in the argument depended on it, and the averaging window is now justified by
# the invariance of beta across windows instead. The computation still lives in
# density_text_numbers.py if it is ever asked for.
# ---------------------------------------------------------------------------

# ============================================================ Tables 7 and 8
bl = stats[(stats["policy_melhorias"] == C.BASELINE_PM)
           & (stats["interest_housing"] == C.BASELINE_IH)
           & (stats["month"] >= EVAL_START) & (stats["month"] <= EVAL_END)].copy()
print("baseline eval rows:", len(bl), "sims:", bl["simulation_id"].nunique(),
      "cities:", bl["processing_acps"].nunique())

VARS = [
    ("Unemployment rate", "unemployment", "{:.4f}"),
    ("Gini coefficient", "gini_index", "{:.4f}"),
    ("Housing vacancy rate", "house_vacancy", "{:.4f}"),
    ("Zero consumption share", "pct_zero_consumption", "{:.4f}"),
    ("GDP growth rate (monthly)", "gdp_growth_rate", "{:.4f}"),
    ("Inflation rate (monthly)", "inflation", "{:.4f}"),
    ("Quality of Life Index", "average_qli", "{:.4f}"),
    (r"Avg.\ house price (model units)", "house_price", "{:.1f}"),
    (r"Avg.\ rent (model units)", "house_rent", "{:.4f}"),
    ("Population", "pop", "{:.1f}"),
    ("Number of domiciles", "number_domiciles", "{:.1f}"),
    ("Loan approval rate", "loan_approval_rate", "{:.4f}"),
    ("Active loans", "active_loans", "{:.1f}"),
    ("Credit stock (model units)", "credit_stock", "{:.1f}"),
    (r"CO$_2$ emissions (model units)", "emissions", "{:.1f}"),
    ("Share firms profit $>0$", "share_firms_positive_profit", "{:.4f}"),
    ("Wage per worker (model units)", "firms_wage_per_worker", "{:.2f}"),
]
n_obs = f"{len(bl):,}"
rows7 = []
for label, col, fmt in VARS:
    s = bl[col].astype(float)
    rows7.append(f"{label} & {fmt.format(s.mean())} & {fmt.format(s.std())} & "
                 f"{fmt.format(s.min())} & {fmt.format(s.median())} & "
                 f"{fmt.format(s.max())} & {n_obs} \\\\")

n_sims = bl["simulation_id"].nunique()
n_cities = bl["processing_acps"].nunique()
t7 = ("\\begin{table}[htbp]\n\\centering\n"
      f"\\caption{{Descriptive statistics of simulated indicators under the baseline "
      f"configuration (media interest rate, no housing improvement policy), evaluation "
      f"window 2011--2021, pooled across {n_sims} simulations "
      f"in {n_cities} cities.}}\n"
      "\\label{tab:descriptive_stats}\n\\resizebox{\\textwidth}{!}{%\n"
      "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
      "Variable & Mean & Std & Min & Median & Max & N \\\\\n\\midrule\n"
      + "\n".join(rows7) +
      "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}")
(TABLES / "Table6_descriptive_stats.tex").write_text(t7)
print("wrote Table7")

CITY_TEX = {
    "RIO DE JANEIRO": "Rio de Janeiro", "BELO HORIZONTE": "Belo Horizonte",
    "PORTO ALEGRE": "Porto Alegre", "RECIFE": "Recife", "BRASILIA": "Brasilia",
    "SALVADOR": "Salvador", "FORTALEZA": "Fortaleza", "CURITIBA": "Curitiba",
    "GOIANIA": "Goiania", "BELEM": "Belem", "MANAUS": "Manaus",
    "VITORIA": "Vitoria", "SAO LUIS": "Sao Luis", "NATAL": "Natal",
    "MACEIO": "Maceio", "TERESINA": "Teresina", "FLORIANOPOLIS": "Florianopolis",
    "JOAO PESSOA": "Joao Pessoa", "ARACAJU": "Aracaju", "CUIABA": "Cuiaba",
    "CAMPO GRANDE": "Campo Grande", "MACAPA": "Macapa",
    "PORTO VELHO": "Porto Velho", "RIO BRANCO": "Rio Branco",
    "BOA VISTA": "Boa Vista", "PALMAS": "Palmas",
}
cty = (bl.groupby("processing_acps")
       .agg(pop=("pop", "mean"), unemp=("unemployment", "mean"),
            gini=("gini_index", "mean"), vac=("house_vacancy", "mean"),
            zc=("pct_zero_consumption", "mean"), qli=("average_qli", "mean"),
            la=("loan_approval_rate", "mean"))
       .sort_values("pop", ascending=False))
rows8 = [f"{CITY_TEX.get(c, c.title())} & {r['pop'] / 1000:.0f} & {r['unemp']:.3f} & "
         f"{r['gini']:.3f} & {r['vac']:.3f} & {r['zc']:.3f} & {r['qli']:.3f} & "
         f"{r['la']:.3f} \\\\"
         for c, r in cty.iterrows()]
m = cty.mean()
mean_row = (f"\\textbf{{Mean}} & {m['pop'] / 1000:.0f} & {m['unemp']:.3f} & "
            f"{m['gini']:.3f} & {m['vac']:.3f} & {m['zc']:.3f} & {m['qli']:.3f} & "
            f"{m['la']:.3f} \\\\")
t8 = ("\\begin{table}[htbp]\n\\centering\n"
      "\\caption{Per-city simulated outcomes, baseline scenario (media interest rate, "
      "no housing improvement policy), evaluation window 2011--2021, averaged over "
      "the city's replications.}\n\\label{tab:city_results}\n"
      "\\resizebox{\\textwidth}{!}{%\n"
      "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
      "City & Pop (k) & Unemp & Gini & Vacancy & ZeroCon & QLI & LoanAppr \\\\\n"
      "\\midrule\n" + "\n".join(rows8) + "\n\\midrule\n" + mean_row +
      "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}")
(TABLES / "Table7_city_results.tex").write_text(t8)
print("wrote Table8")
print(cty.round(4).to_string())
