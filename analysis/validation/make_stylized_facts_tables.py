"""Render Tables 1 and 6 of text_density from the stylized-facts R output.

Input : analysis/validation/stylized_facts_{results,slope_by_cell}_<tag>.csv
        (produced by analysis/validation/stylized_facts.R)
Output: papers/density_housing_inequality/tables/{Table1_phillips_okun_main,
        Table5_phillips_okun_scenarios}.tex

Usage:  python analysis/validation/make_stylized_facts_tables.py new26
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TABLES = ROOT / "papers" / "density_housing_inequality" / "tables"

TAG = sys.argv[1] if len(sys.argv) > 1 else "new26"

# The paper prints Phillips rows first, then Okun; the R table is in that order
# already, so only the display labels are needed.
ROW_LABELS = {
    "Phillips: full panel, contemp.": "Full panel, contemporaneous",
    "Phillips: full panel, city FE + scenario FE": "Full panel, city + scenario FE",
    "Phillips: media/False, full horizon, contemp.":
        "Baseline scenario, full horizon, contemporaneous",
    "Phillips: media/False, full horizon, lagged":
        "Baseline scenario, full horizon, lagged",
    "Okun: full panel": "Full panel",
    "Okun: full panel, city FE + scenario FE": "Full panel, city + scenario FE",
    "Okun: media/False, full horizon": "Baseline scenario, full horizon",
}

CELL_LABELS = {
    "alta_FALSE": "High / Inactive", "alta_TRUE": "High / Active",
    "baixa_FALSE": "Low / Inactive", "baixa_TRUE": "Low / Active",
    "media_FALSE": "Medium / Inactive", "media_TRUE": "Medium / Active",
}
CELL_ORDER = ["alta_FALSE", "alta_TRUE", "baixa_FALSE", "baixa_TRUE",
              "media_FALSE", "media_TRUE"]


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_p(p):
    return r"$<0.001$" if p < 0.001 else f"${p:.4f}$"


def main():
    res = pd.read_csv(HERE / f"stylized_facts_results_{TAG}.csv")
    cel = pd.read_csv(HERE / f"stylized_facts_slope_by_cell_{TAG}.csv")

    # ------------------------------------------------------------- Table 1
    def row(spec, prec):
        r = res[res["spec"] == spec].iloc[0]
        return (f"{ROW_LABELS[spec]} & {r.beta:+.{prec}f}{stars(r.p_value)} "
                f"& {fmt_p(r.p_value)} \\\\")

    phil = [s for s in ROW_LABELS if s.startswith("Phillips")]
    okun = [s for s in ROW_LABELS if s.startswith("Okun")]
    body = ("\n".join(row(s, 4) for s in phil)
            + "[4pt]\n"          # the row already ends in \\, so this makes \\[4pt]
            + r"\multicolumn{3}{l}{\textit{Okun's law} (unemployment on output)} \\[2pt]"
            + "\n" + "\n".join(row(s, 3) for s in okun))

    t1 = r"""\begin{table}[htbp]
\centering
\caption{Phillips curve and Okun's law: baseline specifications}
\label{tab:phillips_okun_main}
\begin{threeparttable}
\begin{tabular}{lcc}
\toprule
Specification & $\hat{\beta}$ & $p$-value \\
\midrule
\multicolumn{3}{l}{\textit{Phillips curve} (prices on unemployment)} \\[2pt]
""" + body + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item Notes: panel regressions on monthly, city-level simulated series with metropolitan-area fixed effects and standard errors clustered by metropolitan area. The Phillips specification regresses monthly inflation on the unemployment rate; the Okun specification regresses monthly GDP growth on the month-on-month change in unemployment, first-differenced within each simulation so no difference is taken across runs. ``Full panel'' pools all six policy configurations, ``baseline scenario'' restricts to the reference configuration (medium interest rate, improvement policy inactive). ``City + scenario FE'' adds fixed effects for the interest-rate $\times$ improvement-policy cell; ``lagged'' uses $X_{t-1}$ in place of $X_{t}$.
\item The first 18 months of every run are dropped as burn-in: unemployment falls steeply from its initialisation value over that transient, and including it dominates the estimated relationships.
\item Both relationships are negative and significant in every specification, consistent with the target stylised facts described in Section~\ref{sec:calibration}.
\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$.
\end{tablenotes}
\end{threeparttable}
\end{table}"""
    (TABLES / "Table1_phillips_okun_main.tex").write_text(t1)
    print("wrote Table1")
    print(res.to_string(index=False))

    # ------------------------------------------------------------- Table 6
    lines = []
    # Phillips coefficients are O(1e-3) and their SEs O(1e-4); at the four
    # decimals the published table used, every SE would print as 0.0001/0.0002.
    for test, label, prec in [("Okun", "Okun's law", 3),
                              ("Phillips", "Phillips curve", 5)]:
        sub = cel[cel["test"] == test].set_index("scenario_cell")
        for j, cell in enumerate(CELL_ORDER):
            r = sub.loc[cell]
            lines.append(
                f"{label} & {CELL_LABELS[cell]} & {r.beta:+.{prec}f}{stars(r.p_value)} "
                f"& {r.std_error:.{prec}f} & {fmt_p(r.p_value)} \\\\"
                + (r"[4pt]" if test == "Okun" and j == len(CELL_ORDER) - 1 else ""))

    n_all = len(cel)
    n_01 = int((cel["p_value"] < 0.01).sum())
    n_05 = int((cel["p_value"] < 0.05).sum())
    n_neg = int((cel["beta"] < 0).sum())
    assert n_neg == n_all, "a scenario-level estimate flipped sign; rewrite the note"
    word = {12: "twelve", 10: "ten", 9: "nine", 8: "eight", 7: "seven", 6: "six"}

    t6 = r"""\begin{table}[htbp]
\centering
\caption{Phillips curve and Okun's law: stability across the six policy scenarios}
\label{tab:phillips_okun_scenarios}
\begin{threeparttable}
\begin{tabular}{llccc}
\toprule
Test & Scenario (rate / improvement policy) & $\hat{\beta}$ & Std.\ error & $p$-value \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item Notes: same estimating equation as Table~\ref{tab:phillips_okun_main}, with the slope estimated separately within each of the six policy scenarios (regulated credit rate $\times$ improvement policy status) rather than pooled, as a stability check on the sign and significance of both stylised facts.
\item Every one of the """ + word.get(n_all, str(n_all)) + r""" scenario-level estimates is negative and significant at $p<0.05$ (""" + word.get(n_01, str(n_01)) + r""" of the """ + word.get(n_all, str(n_all)) + r""" at $p<0.01$), indicating the macroeconomic regularities targeted in calibration hold within each policy configuration individually, not only in the pooled panel.
\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$.
\end{tablenotes}
\end{threeparttable}
\end{table}"""
    assert n_05 == n_all, "a scenario-level estimate lost significance; rewrite the note"
    (TABLES / "Table5_phillips_okun_scenarios.tex").write_text(t6)
    print("wrote Table6")
    print(cel.to_string(index=False))


if __name__ == "__main__":
    main()
