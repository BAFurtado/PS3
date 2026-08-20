# Housing policy and urban inequality across 27 Brazilian metropolitan areas

Reproduction package for the PolicySpace3 density paper.

The manuscript source is maintained on Overleaf; this directory holds everything needed to
rebuild the numbers in it — the analysis-ready data, the pipeline that produced it, the
figures and tables it generates, and the submitted PDF as the reference of record.

---

## What you can reproduce, and what it costs

| | Needs | Rebuilds |
|---|---|---|
| **Nothing to download** | `data/` in this directory | Figures 2–6, all regression results (`regression.py` §1–§10, `regression_table.tex`) |
| **`final_stats` (682 MB)** | + the download below | Figure 1, Tables 1, 4 and 5, the ΔGini monthly series |
| **`+ regional_stats` (1.7 GB)** | + a second file, not yet uploaded | Tables 3, 6 and 7 (municipality, capital and periphery levels) |
| **`+ output/sensitivity/`** | + the OAT batch | Table 8 |

The derived panels in `data/` are committed precisely so the first row needs no download at
all. Start there.

## The simulated database

The paper rests on **1,596 simulations**: 26 metropolitan areas × 6 policy configurations ×
10 seeds, plus São Paulo × 6 configurations × 6 seeds, each over a 30-year horizon
(2010-01 to 2039-12) at 1 % population sampling.

> **`final_stats_27cities.csv` (682 MB)** —
> https://drive.google.com/file/d/1E6XFjAIaKPq3ETRLQuDCW0lVWdhc8uyj/view?usp=drive_link

Place it at `output/final_stats_27cities.csv`, relative to the PolicySpace3 project root.
Verify you have the right batch: it should carry **27 distinct `processing_acps` values and
1,596 distinct `simulation_id` values**.

Three further inputs are not yet published; ask the corresponding author if you need them.

| Path | Size | Needed for |
|---|---|---|
| `output/regional_stats_27cities.csv` | 1.7 GB | municipality-level panels → Tables 3, 6, 7 |
| `output/sensitivity/` | 368 MB | Table 8 |
| `output/sensitivity_goiania_melhorias/` | 75 MB | Table 8, Goiânia *melhorias* arm |

### Regenerating the batch from scratch

On a machine with the PolicySpace3 environment (see the root `README.md`):

```bash
python tests.py                      # expect all PASS; do not launch on a failure
rm -f StoragedAgents/*.agents        # pickled agent classes must match the current code

# 25 capitals: 25 x 6 configs x 10 seeds = 1,500 runs (~2-3 days on 40 cores)
python main.py -n 10 -c 40 sensitivity PLANHAB-capitais

# Sao Paulo and Rio are run separately: together they cost as much as the other 25
python main.py -n 10 -c 12 sensitivity PLANHAB-SAO_PAULO
python main.py -n 10 -c 12 sensitivity PLANHAB-RIO_DE_JANEIRO
```

Set `-c` to the number of concurrent runs you are giving it; a lower `-c` does not drop runs,
it roughly doubles wall-clock by forcing a second wave. Use no `-p`/`-r` overrides — the
defaults for `TOTAL_DAYS`, `SAVE_DATA` and `QLI_TAX_WEIGHT` already match the paper's design,
and any override breaks comparability.

Then concatenate the per-launch `stats` and `regional` outputs into
`output/final_stats_27cities.csv` and `output/regional_stats_27cities.csv`. The launch
timestamps inside `simulation_id` must survive the concatenation: the seed-matching key in
`prepare_data.py` §3 depends on them.

Check the batch before trusting it:

```bash
python analysis/planhab/check_smoke_batch.py output/<TIMESTAMP>
```

Two further checks are worth running by hand. Every configuration of a replication must
record the same `PARAMS.SEED` in its `conf.json`; and because the three `INTEREST_HOUSING`
scenarios share historical rates before 2020-01, runs with the same `POLICY_MELHORIAS` and
seed must be identical up to 2019-12. Any pre-2020 divergence means the seeds are not matched.

---

## Pipeline

Run every script from the **PolicySpace3 project root**, not from this directory.

| Step | Command | Produces | Needs raw batch |
|---|---|---|---|
| 1. Build the panels | `python papers/density_housing_inequality/prepare_data.py` | `data/{acp,mun,capital,periphery}_deltas.csv`, `data/acp_population.csv`, `data/capital_mun_ids.csv` | yes, both files |
| 2. Regressions | `python papers/density_housing_inequality/regression.py` | `data/regression_*.csv`, `data/regression_table.tex`, `data/timeseries_*.csv` | §11 only |
| 3. Figures 2–6 | `python papers/density_housing_inequality/make_figures.py` | `fig_*.pdf` / `fig_*.png` alongside the script | no |
| 4. Window robustness | `python papers/density_housing_inequality/robustness_window.py` | `data/robustness_{window,stationarity}_table.tex` | yes |

`make_figures.py` writes `fig_<name>.pdf`, while the manuscript includes
`figures/Figure<n>_<name>`. The rename is not automated; apply it after step 3:

```bash
cd papers/density_housing_inequality
mv fig_timeseries.pdf           figures/Figure2_timeseries.pdf
mv fig_mechanism_scatter.pdf    figures/Figure3_mechanism_scatter.pdf
mv fig_appendix_ranked_bars.pdf figures/Figure4_appendix_ranked_bars.pdf
mv fig_appendix_violin.pdf      figures/Figure5_appendix_violin.pdf
mv fig_appendix_pop_scatter.pdf figures/Figure6_appendix_pop_scatter.pdf
rm -f fig_*.png
```

Figure 1 comes from `validation_article.py` and is not touched by this step.

The tables and Figure 1 are produced by scripts in the main analysis tree, which write
directly into this directory's `tables/`:

| Output | Script |
|---|---|
| Figure 1 (normalised time-series validation) | `analysis/validation/validation_article.py` |
| Tables 1 and 5 (Phillips / Okun) | `analysis/validation/stylized_facts.R`, then `analysis/validation/make_stylized_facts_tables.py new26` |
| Tables 3, 4, 6, 7 | `analysis/validation/regen_tables.py` |
| Table 8 (parameter sensitivity) | `analysis/validation/density_sensitivity_numbers.py output/sensitivity` |
| In-text numbers | `analysis/validation/density_text_numbers.py` |

`stylized_facts.R` is R, not Python; `make_stylized_facts_tables.py` renders its CSV output
(`analysis/validation/stylized_facts_{results,slope_by_cell}_new26.csv`, committed) into LaTeX.

Figure 1 additionally reads empirical series from
`analysis/validation/real_world_data/real_data_macroeconomic.csv` and
`analysis/validation/real_world_data/mun_isic12_2010.csv`, both committed.

---

## Reproduction check

Steps 2 and 3 were re-run from this package on 2026-08-20 against the raw batch. Figures 2–6
match the committed PDFs, and `data/regression_table.tex` and the panel CSVs are byte-identical
to the versions behind the submitted tables. Three of the regression CSVs differ only in the
fifteenth significant digit of `r2` and `se`, which is BLAS summation-order noise: every
coefficient, standard error, p-value and significance star is unchanged.

---

## Contents

| | |
|---|---|
| `data/` | the analysis-ready panels: 27 cities, 1,330 delta observations |
| `figures/` | the six figures, as included by the manuscript |
| `tables/` | the seven tables, as `\input`-ed by the manuscript |
| `manuscript_submitted.pdf` | the compiled paper as submitted |
| `*.py` | the four pipeline scripts |

The manuscript source lives on Overleaf and is not mirrored here; `manuscript_submitted.pdf`
is the reference of record for what the numbers in this package correspond to.

The model itself, its parameters and its input data are documented in the repository root
`README.md`; the empirical provenance of every input dataset is in `input/data_sources.md`.
