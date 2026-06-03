# PolicySpace3 (PS3)

![Python](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/license-GNU%20v3-green)

**An open agent-based platform for ex ante simulation of public policies in Brazilian urban areas.**

PS3 is a computational simulation platform designed to support prospective analysis of public policies. It models heterogeneous agents — families, firms, local governments, and a financial institution — interacting across multiple markets (labor, goods and services, housing, rental, and mortgage credit) within spatially detailed Brazilian urban concentration areas (ACPs/CONURBs).

Rather than estimating relationships observed in the past, PS3 enables exploration of alternative scenarios, institutional changes, and policy combinations whose outcomes are not yet known. The platform combines census microdata, administrative records, and official statistics to represent the evolution of complex urban systems over time.

---

## Applications

| Theme | Reference |
|-------|-----------|
| Carbon taxes, green subsidies, and emissions | Rocha Lima, Furtado & Lopes (2026) |
| National Housing Plan (PlanHab) scenarios | Ongoing / institutional reports |
| Urban mobility (commute times) | Ongoing |
| Housing markets and endogenous public policies (PolicySpace2) | Furtado (2022a, 2022b) |
| Platform documentation (v1 — PolicySpace) | Furtado (2018a, 2018b) |
| Metropolitan governance and fiscal redistribution | Alves Furtado & Eberhardt (2016); Furtado, Eberhardt & Messa (2017) |

---

## Architecture

### Agents
- **Families**: supply labor, consume goods and services, accumulate wealth, demand housing, and make residential relocation decisions
- **Firms**: produce goods and services across **12 economic sectors** (Agriculture, Mining, Manufacturing, Utilities, Construction, Trade, Transport, Business, Financial, Real Estate, Other Services, Government); hire workers and set prices using input-output technical coefficients
- **Local governments**: collect taxes (consumption, labor, property) and implement public policies (housing subsidies, MCMV, FGTS, SBPE, PlanHab)
- **Bank**: manages mortgage loans, deposits, and interest rates (FGTS and SBPE funding lines)

### Markets
- **Labor market**: job matching by distance and qualification, wage determination, hiring/firing
- **Goods and services market**: consumption, intermediate production, regional input-output flows with external suppliers
- **Housing sale market**: price formation with submarkets segmented by income quartile
- **Rental market**: rental pricing, payment, and default
- **Mortgage credit market**: loan applications, approvals, repayments, and interest rates

### Spatial structure
Simulations run for Brazilian urban concentration areas (ACPs/CONURBs). The basic spatial unit is the **IBGE weighting area (área de ponderação / AP)**, with municipal boundaries used as fallback where AP shapes are unavailable. Regionalized input-output matrices capture inter-sectoral production relationships, including inputs acquired from suppliers outside the simulated area.

### Demographic dynamics
Population evolves endogenously through births, deaths (age- and state-specific mortality tables), aging, immigration, marriage, and household formation and dissolution. Government employment follows exogenous projections by year.

### Output
Each simulation produces time-series indicators saved as CSV files, covering unemployment, inequality (Gini index), housing vacancy and prices, fiscal revenues, consumption, credit, and environmental emissions. Results are automatically plotted with confidence intervals across multiple runs.

An **interactive dashboard** for exploring simulation results is available at: [link — coming soon]. To use it, point the dashboard to a `final_stats.csv` produced by any completed run.

---

## Installation

Requires **Python 3.12**. We recommend `conda` to create an environment with all spatial and scientific dependencies at once:

```bash
conda create -n ps3 -c conda-forge \
  fiona geopandas shapely gdal pandas numba descartes scipy seaborn \
  pyproj matplotlib six cycler statsmodels joblib scikit-learn \
  flask flask-wtf psutil pyarrow
```

Then add the financial mathematics library:

```bash
pip install numpy_financial
```

Clone the repository:

```bash
git clone https://github.com/BAFurtado/PS3.git
cd PS3
```

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
`-n`: number of runs | `-c`: CPU cores (`-1` for all available). Multiple runs produce averages and confidence intervals.

### Sensitivity analysis
```bash
# Continuous parameter range (name:min:max:steps)
python main.py -n 2 sensitivity ALPHA:0:1:7

# Boolean parameter (runs True and False)
python main.py -n 2 sensitivity SOME_BOOLEAN_PARAM

# Two parameters with specific value combinations
python main.py -n 2 sensitivity MARKUP:.05:.15:7 PRODUCTIVITY_EXPONENT:.4:.6:3
```

### PlanHab housing policy scenarios
```bash
python main.py sensitivity PLANHAB-capitais
python main.py sensitivity PLANHAB-BELO-HORIZONTE-SAO-PAULO
```

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

### Run tests
```bash
python tests.py
```

---

## Configuration

Override defaults without modifying core files by creating:

- `conf/params.py` — simulation parameters (e.g., `PRICE_MARKUP_CAP`, `LABOR_MARKET`)
- `conf/run.py` — run options (e.g., `OUTPUT_PATH`, `SAVE_DATA_PERIDIOCITY`, `AVERAGE_TYPE`)

All defaults are documented in `conf/default/params.py` and `conf/default/run.py`.

You can also pass JSON overrides directly on the command line:

```bash
python main.py -p custom_params.json -r custom_config.json run
```

---

## Collaborators

**Current**
- Gustavo Libório Rocha Lima
- João Victor Lisbôa de Vasconcelos

**Earlier contributions**
- Isaque Daniel Rocha Eberhardt
- Francis Tseng — [frnsys.com](http://frnsys.com)

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

Developed by Bernardo Alves Furtado, funded primarily by the Institute of Applied Economic Research — Ipea ([www.ipea.gov.br](https://www.ipea.gov.br)). Partial support from CEPAL-Brasília and the International Policy Centre (IPC-IG). Bernardo Alves Furtado acknowledges a CNPq productivity grant (2014–2023).

## License

GNU General Public License v3.0