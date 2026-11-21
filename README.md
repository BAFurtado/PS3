# PolicySpace3 (PS3)

![Python](https://img.shields.io/badge/python-3.12-blue) ![GitHub](https://img.shields.io/github/license/bafurtado/PS3) ![GitHub stars](https://img.shields.io/github/stars/bafurtado/PS3.svg?color=orange) ![GitHub forks](https://img.shields.io/github/forks/bafurtado/PS3)

**An open agent-based platform for ex ante simulation of public policies in Brazilian urban areas.**

This release accompanies the paper *More Social Housing, More and Smaller Families: PolicySpace3, an Agent-Based Model of Brazilian Metropolitan Economies* (Furtado, Rocha Lima and Vasconcelos). The first section below explains how to reproduce the paper's runs, figures and tables. The sections after it describe the platform.

## Reproducing the paper

### Model documentation

The Supplementary Information of the paper (`docs/ps3_model_si.pdf`) describes the model with the ODD protocol, gives every parameter with its value and source, and reports the validation tables and the sensitivity analysis.

### Installation

Python 3.12. Clone the repository and create the environment with conda:

```bash
git clone https://github.com/BAFurtado/PS3.git
cd PS3
conda create -n ps3 -c conda-forge python=3.12 \
  geopandas shapely pandas numpy scipy matplotlib statsmodels scikit-learn \
  joblib click pyarrow requests tqdm python-dateutil salib
pip install numpy_financial
```

Or with pip: `pip install -r requirements.txt`.

Run every command below from the repository root. The input data are in `input/`. The first run of an urban concentration area builds its initial population from the 2010 Census and stores it in `StoragedAgents/`, and later runs reuse it. Plotting scripts need `MPLBACKEND=Agg` on a machine without a display.

### Tests

```bash
python tests.py
```

### The policy runs

`runner_mcmv2026.py` runs the two MCMV budgets of the paper: the reference, `d09`, with 0.09 % of GDP a year from 2026, and the intervention, `d25`, with 0.25 % in 2026–2030 and 0.09 % from 2031. It also runs `d13`, with 0.13 % in 2026–2030, which the paper does not report. Each urban concentration area runs ten seeds, 2010–2034. The runner sets every scenario parameter and ignores a local `conf/params.py` or `conf/run.py`.

```bash
python runner_mcmv2026.py teste 3                  # Palmas, 2 years, every dose, then the identity check
python runner_mcmv2026.py lote 8                   # the 26 capitals other than São Paulo, 8 runs at a time
python runner_mcmv2026.py lote 8 cities=sp         # São Paulo, d09 and d25 (no agents parquet)
python runner_mcmv2026.py lote 8 part=1/2          # the first of two slices of the queue, one per server
python runner_mcmv2026.py conferir                 # finished runs, and d13 and d25 equal to d09 before 2026
```

Rerun the same `lote` command to resume: finished runs are skipped. Output goes to `output/mcmv2026/<scenario>/<CITY>/<seed>/`.

A run of a mid-sized urban concentration area to 2034 takes about 20 to 30 minutes on one core. A São Paulo run takes 6 to 12 hours and about 2.5 GB of RAM.

Full targeting (Goiânia, the poorest eligible households first) runs with

```bash
python analysis/mcmv2026/queue_test.py 8 0 9      # seeds 0 to 9, into output/queue_test
python analysis/mcmv2026/queue_compare.py 9
```

### Calibration and sensitivity analysis

```bash
python -m analysis.calibration.sample run-sample --samples 32 --cpus 8     # history matching, one wave
python -m analysis.calibration.sample score <calibration output folder>

python runner_morris.py lote 8           # Morris screen, Goiânia and Aracaju (design in analysis/morris)
python analysis/morris/analyse.py        # Supplementary Information, Table B.3
python runner_morris_ends.py lote 8      # each influential parameter at the ends of its range
python analysis/morris/ends.py           # Supplementary Information, Table B.4 (reads output/mcmv2026)
```

### Analysis, figures and tables

After the runs, from the repository root, in this order:

```bash
python analysis/mcmv2026/effects.py output/mcmv2026 6
python analysis/mcmv2026/paths.py
python analysis/mcmv2026/screen.py
python analysis/mcmv2026/price_composition.py
python analysis/mcmv2026/hetero_extract.py output/mcmv2026 6
python analysis/mcmv2026/hetero_analysis.py
python analysis/mcmv2026/yearly_extract.py output/mcmv2026 6
python analysis/mcmv2026/city_traits.py
python analysis/mcmv2026/households_census.py
python analysis/mcmv2026/households_census_mun.py
python analysis/mcmv2026/excess_burden.py
python analysis/mcmv2026/shortfall.py
python analysis/mcmv2026/validation/screen25.py output/mcmv2026/d09 output/mcmv2026/_analysis/screen25_27
python analysis/mcmv2026/fig_validation.py
python analysis/mcmv2026/figures.py
```

The scripts write to `output/mcmv2026/_analysis/`. Each script's docstring names its inputs and outputs.

| Paper | Script |
|---|---|
| Figure 2, budget | `figures.py` (`fig_budget`) |
| Figure 3 and Table B.1, validation | `validation/screen25.py`, `fig_validation.py` |
| Figures 4 to 6, effects over time, by place and income, by metropolitan area | `figures.py` (`fig_paths`, `fig_whom`, `fig_cities`) |
| Effects, ledger and prices in Section 6 | `effects.py`, `paths.py`, `screen.py`, `price_composition.py`, `hetero_analysis.py`, `yearly_extract.py`, `excess_burden.py`, `city_traits.py` |
| Population and households against the Census, Table B.2 | `shortfall.py`, `households_census.py`, `households_census_mun.py` |

`households_census.py` and `households_census_mun.py` read two IBGE Sidra tables saved as `output/mcmv2026/_analysis/sidra/c2010.json` (https://apisidra.ibge.gov.br/values/t/156/n6/all/v/2048,134/p/2010) and `c2022.json` (https://apisidra.ibge.gov.br/values/t/4712/n6/all/v/381,382/p/2022).

### Licence

MIT, see `LICENSE`.

---

## About the platform

PS3 is a computational simulation platform for the prospective analysis of public policies. It models heterogeneous agents (families, firms, local governments and a financial institution) that interact in labour, goods and services, housing, rental and mortgage credit markets within spatially detailed Brazilian urban concentration areas (ACPs/CONURBs).

PS3 simulates alternative scenarios, institutional changes and policy combinations whose outcomes are not yet known. The platform combines census microdata, administrative records and official statistics to represent the evolution of urban systems over time.

---

## Applications

| Theme | Reference |
|-------|-----------|
| Carbon taxes, green subsidies, and emissions | Rocha Lima, Furtado & Lopes (2026) |
| National Housing Plan (PlanHab) scenarios | Ongoing / institutional reports |
| Urban mobility (commute times) | Ongoing |
| Housing markets and endogenous public policies (PolicySpace2) | Furtado (2022a, 2022b) |
| Platform documentation (v1, PolicySpace) | Furtado (2018a, 2018b) |
| Metropolitan governance and fiscal redistribution | Alves Furtado & Eberhardt (2016), Furtado, Eberhardt & Messa (2017) |

---

## Architecture

### Agents
- **Families** supply labour, consume goods and services, accumulate wealth, demand housing and decide whether and where to move.
- **Firms** produce goods and services in **12 economic sectors** (Agriculture, Mining, Manufacturing, Utilities, Construction, Trade, Transport, Business, Financial, Real Estate, Other Services, Government) with input-output technical coefficients, hire workers and set prices.
- **Local governments** collect taxes (consumption, labour, property) and run public policies (housing subsidies, MCMV, FGTS, SBPE, PlanHab).
- **The bank** manages mortgage loans, deposits and interest rates (FGTS and SBPE funding lines).

### Markets
- The **labour market** matches workers to jobs by distance, sets wages, and hires and fires.
- The **goods and services market** covers consumption, intermediate production and regional input-output flows with suppliers outside the area.
- The **housing sale market** forms prices in submarkets segmented by income quartile.
- The **rental market** sets rents and handles payment and default.
- The **mortgage credit market** handles loan applications, approvals, repayments and interest rates.

### Spatial structure
Simulations run for Brazilian urban concentration areas (ACPs/CONURBs). The basic spatial unit is the **IBGE weighting area (área de ponderação, AP)**, and municipal boundaries replace it where AP shapes are unavailable. Regionalised input-output matrices represent the flows between sectors, including inputs bought from suppliers outside the simulated area.

### Demographic dynamics
Population evolves endogenously through births, deaths (age- and state-specific mortality tables), ageing, immigration, marriage, and household formation and dissolution. Government employment follows exogenous projections by year.

### Output
Each simulation saves time series of indicators as CSV files, covering unemployment, inequality (Gini index), housing vacancy and prices, fiscal revenues, consumption, credit and emissions. The model plots the results with confidence intervals across runs.

---

## Running the model

### Single run (default parameters)
```bash
python main.py run
```

### Multiple parallel runs
```bash
python main.py -c 2 -n 10 run
```
`-n` sets the number of runs and `-c` the number of CPU cores (`-1` for all). Several runs give averages and confidence intervals.

### Sensitivity analysis
```bash
# A continuous range, NAME:MIN:MAX:STEPS
python main.py -n 2 sensitivity MARKUP:.05:.15:7

# A boolean parameter, True and False
python main.py -n 2 sensitivity POLICY_MELHORIAS

# Two parameters, one at a time
python main.py -n 2 sensitivity MARKUP:.05:.15:7 PRODUCTIVITY_EXPONENT:.4:.6:3

# Every combination of the listed values of two parameters, NAME1+NAME2*VALUES1*VALUES2
python main.py -n 2 sensitivity "MARKUP+PRODUCTIVITY_EXPONENT*.05+.1+.15*.4+.6"
```

### PlanHab housing policy scenarios
```bash
python main.py sensitivity PLANHAB-capitais
python main.py sensitivity "PLANHAB-BELO HORIZONTE-SAO PAULO"
```

Each area runs with and without the Melhorias policy at three housing interest-rate scenarios (`baixa`, `media`, `alta`). `capitais` runs the capitals other than São Paulo and Rio de Janeiro (25 areas). Area names keep their spaces, and a hyphen separates them.

### Run across all urban areas (ACPs)
```bash
python main.py -n 2 -c 2 acps
```

### Resume an interrupted run
```bash
python main.py resume /path/to/output
```

### Regenerate plots from saved data
```bash
python main.py make-plots /path/to/output
```

---

## Configuration

Override defaults without changing the default files by creating:

- `conf/params.py` for simulation parameters (for example `PRICE_MARKUP_CAP`, `LABOR_MARKET`)
- `conf/run.py` for run options (for example `OUTPUT_PATH`, `SAVE_DATA_PERIDIOCITY`, `AVERAGE_TYPE`)

`conf/default/params.py` and `conf/default/run.py` document every default.

The command line also takes overrides, `-p` with a JSON file of parameters and `-r` with a JSON string of run options.

```bash
python main.py -p custom_params.json -r '{"OUTPUT_PATH": "output_test"}' run
```

---
## Collaborators
**PI**
- Bernardo Alves Furtado 

**Current**
- Gustavo Libório Rocha Lima
- João Victor Lisbôa de Vasconcelos

**Earlier contributions**
- Isaque Daniel Rocha Eberhardt
- Francis Tseng ([frnsys.com](http://frnsys.com))

---

## References

Rocha Lima, G. L.; Furtado, B. A.; Lopes, O. F. Innovation or contraction? Unpacking the effects of carbon taxes and subsidies on emissions in Brazil. *Energy Policy*, v. 211, p. 115095, 2026.

Furtado, B. A. PolicySpace2: modeling markets and endogenous public policies. *Journal of Artificial Societies and Social Simulation*, v. 25, n. 1, p. 8, 2022a.

Furtado, B. A. *PolicySpace2: modelando mercado imobiliário e políticas públicas*. Rio de Janeiro: Ipea, 2022b.

Furtado, B. A. PolicySpace: a modeling platform. *Journal on Policy and Complex Systems*, v. 4, n. 2, p. 17–30, 2018a.

Furtado, B. A. *PolicySpace: modelagem baseada em agentes*. Brasília: Ipea, 2018b.

Furtado, B. A.; Eberhardt, I. D. R.; Messa, A. Governance for smart cities: a spatially-bounded economic agent-based lab (SEAL) to foster urban analysis and policy evaluation. In: Aijaz, R. (ed.). *Smart Cities Movement in BRICS*. New Delhi: Observer Research Foundation, 2017. p. 63–71.

Alves Furtado, B.; Eberhardt, I. D. R. A simple agent-based spatial model of the economy: tools for policy. *Journal of Artificial Societies and Social Simulation*, v. 19, n. 4, p. 12, 2016.

---

## Funding

Developed by Bernardo Alves Furtado, funded primarily by the Institute of Applied Economic Research (Ipea, [www.ipea.gov.br](https://www.ipea.gov.br)). Partial support from CEPAL-Brasília and the International Policy Centre (IPC-IG). Bernardo Alves Furtado acknowledges a CNPq productivity grant (2014–2023).

## License

MIT, see `LICENSE`.