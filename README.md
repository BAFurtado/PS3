## PolicySpace3--regional input-output matrices, environmental externalities, exogenous demographic projections
### Applications: PlanHab scenarios, Emissions analysis, Transport policy comparisons, Causality

### Latest Publications

#### Published Papers

1. **Innovation or contraction? Unpacking the effects of carbon taxes and subsidies on emissions in Brazil**
   - *Energy Policy*, Volume 211, April 2026, 115095
   - Authors: Gustavo L. Rocha Lima, Bernardo A. Furtado, Olandia F. Lopes
   - DOI: [10.1016/j.enpol.2026.115095](https://doi.org/10.1016/j.enpol.2026.115095)
   - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0301421526000297)
   - **Key Findings:**
     - Carbon taxes reduce emissions mainly through economic contraction, not efficiency gains
     - Investment subsidies promote innovation but have limited effects on total emissions
     - Combining carbon taxes and subsidies does not create strong environmental synergies
     - Firm-level interactions explain why green policies increase market concentration (Matthew effect)
     - Targeted subsidies to high-emission sectors and tax revenue recycling mitigate inequality without weakening emission reductions

2. **A provocation on Agent-Based Models as causal inference tools**
   - Preprint available at: [SURFdrive](https://surfdrive.surf.nl/files/index.php/s/AUt2TsUOIVb51Zx)

#### Working Papers & Preprints

3. **Optimal policy, which, where, and why**
   - Available at: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4276602)

4. **Machine Learning Simulates Agent-Based Model Towards Policy**
   - Available at: [arXiv](https://arxiv.org/abs/2203.02576)

------
``` python 3.12```
------
------
Previous work below
------

### Collaborators
Bernardo Alves Furtado -- https://sites.google.com/view/bernardo-alves-furtado

Gustavo Liborio Rocha Lima

Francis Tseng --  http://frnsys.com

Isaque Eberhardt

![GitHub labels](https://img.shields.io/github/labels/atom/atom/help-wanted)

### Funding

Developed by Bernardo Alves Furtado, funded primarily by Institute of Applied Economic Research (IPEA) 
[www.ipea.gov.br] with one-period grant from [https://www.cepal.org/pt-br/sedes-e-escritorios/cepal-brasilia] 
(CEPAL-Brasília) and International Policy Centre (https://ipcig.org/). 
BAF acknowledges receiving a grant of productivity by National Council of Research (CNPq) [www.cnpq.br].

#### How do I get set up?

We recommend using conda  and creating an environment that includes all libraries simultaneously.

You do not need to inform the python version. Let the configuration of current libraries requirements decide.

The line below set on `python 3.12` on February 4, 2025-

`conda create -n ps3_2 -c conda-forge fiona geopandas shapely gdal pandas numba descartes scipy seaborn pyproj matplotlib 
six cycler statsmodels joblib scikit-learn flask flask-wtf psutil pyarrow`

Finally, also add numpy_financial
`pip install numpy_financial`

## How to run the model ##

1. `git clone https://github.com/BAFurtado/PS3.git` ou
2. `git fork https://github.com/BAFurtado/PS3.git`

### Configuration

To locally configure the simulation's parameters, create the following files as needed:

- `conf/run.py` for run-specific options, e.g. `OUTPUT_PATH` for where sim results are saved
- `conf/params.py` for simulation parameters, e.g. `LABOR_MARKET`.

The default options are in `conf/default/`, refer to those for what values can be set.

### Parallelization and multiple runs

These optional arguments are available for all the run commands:

- `-n` or `--runs` to specify how many times to run the simulation.
- `-c` or `--cpus` to specify number of CPUs to use when running multiple times. Use `-1` for all cores (default).

### Running

```
python main.py run
```

Example with multiple runs and CPUs:

```
python main.py -c 2 -n 10 run
```

#### Configuration Options

You can also pass configuration overrides via JSON files:

- `-p, --params` - JSON file for parameter overrides (applied per-run)
- `-r, --config` - JSON file for run config overrides (applied globally)

Example:
```
python main.py -p custom_params.json -r custom_config.json run
```

#### Running tests

A few tests are implemented in `tests.py`. You may run them as: 
```
python tests.py
```

#### Sensitivity analysis

Runs simulation over a range of values for specific parameters. To see comparative resulting graphs, 
run at least 2 runs for each parameter (`-n 2` or more).

**Syntax options:**

1. **Continuous parameters**: `NAME:MIN:MAX:STEPS`
   ```
   python main.py -n 2 sensitivity ALPHA:0:1:7
   ```
   Runs for values: `ALPHA=0`, `ALPHA=0.17`, `ALPHA=0.33`, ... `ALPHA=1`

2. **Boolean parameters**: Just provide the parameter name (runs with True/False)
   ```
   python main.py -n 2 sensitivity SOME_BOOLEAN_PARAM
   ```

3. **Multiple parameters with specific values**: Use `+` to combine params and `*` for values
   ```
   python main.py -n 2 sensitivity MARKUP:.05:.15:7 PRODUCTIVITY_EXPONENT:.4:.6:3
   ```
   Or with combinations:
   ```
   python main.py sensitivity PRODUCTIVITY_EXPONENT+PRODUCTIVITY_MAGNITUDE_DIVISOR*.3+.4*10+20
   ```

4. **Select specific ACPs**: Use `- ACP1-ACP2-ACP3` syntax
   ```
   python main.py -n 2 sensitivity -BELO-HORIZONTE-CAMPINAS
   ```

5. **PlanHab scenarios**: Use `PLANHAB-capitais` or specify cities
   ```
   python main.py sensitivity PLANHAB-capitais
   python main.py sensitivity PLANHAB-BELO-HORIZONTE-SAO-PAULO
   ```

#### Distributions

Runs simulation over different distribution combinations: `ALTERNATIVE0` and `FPM_DISTRIBUTION` (True/False).

```
python main.py -n 2 -c 2 distributions
```

#### Distributions across ACPs

Runs distributions for multiple ACPs (excludes some single-municipality ACPs):

```
python main.py -n 2 -c 2 distributions_acps
```

#### ACPs

Runs simulation across different Urban Concentration Areas (ACPs/CONURBs):

```
python main.py -n 2 -c 2 acps
```

#### Regenerating plots

You can regenerate plots for an output directory:

```
python main.py make-plots /path/to/output
```

In Windows, use double quotes and backward slashes:
```
python main.py make-plots "..\run__2017-11-01T11_59_59.240250_bh"
```

### Running the web interface

> **Note:** The web interface is currently disabled in the codebase. It was a preliminary development 
> and has not been maintained. To re-enable it, uncomment the relevant lines in `main.py`.
> 
> 
PolicySpace2: Real Estate Modeling
------
<img alt="GitHub stars" src="https://img.shields.io/github/stars/bafurtado/policyspace2.svg?color=orange">  ![GitHub All Releases](https://img.shields.io/github/downloads/bafurtado/policyspace2/total) ![GitHub](https://img.shields.io/github/license/bafurtado/policyspace2)  ![GitHub forks](https://img.shields.io/github/forks/bafurtado/policyspace2)
------

<a href="https://www.comses.net/codebases/c8775158-4360-46d8-bac8-be94502b04b0/releases/1.2.0/"><img src="https://www.comses.net/static/images/icons/open-code-badge.png" align="left" height="64" width="64" ></a>

---
#### Reviewed code at ComSES

## Post-PolicySpace2 work

### Changes and enhancements compared to PolicySpace2
1. Fix regional plotting with confidence intervals
2. Optimization of consume and update policy functions
3. Updated data to run into 2040 (resources)
4. Reorganized policy triggering (to include MCMV faixa 1)
5. Included subsided interest for FGTS and SBPE
6. Gathered data from MCidades for ações MCMV faixa 1, calculated as GDP weighted values
7. Miscelaneous optimization
8. Fix head-rate saving mechanism
9. Modified labour matching for a Cobb-Douglas with 3 parameters
10. Included melhorias (housing improvements) policy
11. Addresses chosen with spatial join (instead of individually). Enhancement: faster run
12. Fixed bug construction firm planning house
13. Introduced new parameter to control for availability of lot space supply for construction
14. Changed output figures to include lower-upper bound confidence intervals, instead of plotting all runs lines.
15. Sales and rental markets now use sub-markets in quartiles by income distributions. 
    Families in the lower income quartile search for houses in the lower quartile of quality
16. Endogenized decision to enter house market: based on employability, renting status, funds available, level consumption
17. Ordered, organized decision on consumption
18. Changed price and production decisions based on Dawid, 2018 benchmarks
19. Checked bank loan systems. 
20. Included new graph reports: wages received, wages paid, firms' stocks, population
21. Sorting of firms in the labor market made by per capita revenue!
22. Introduced parameters PRICE_RUGGEDNESS so that prices reduction is more rugged than when increasing
23. Now there is endogenous qualification growth, observing dropout average magnitude
24. Plotting with lower upper bounds (2std) for sensitivity comparison
25. Introduced test of collecting consumption taxes at consumers' municipality rather than firms'
26. Migrated from using `OGR from OSGEO` to regular `geopandas.DataFrames`
27. Initial real estate average area estimates of empirical values from FIPEZAP in the beginning are read from file, when available
28. Introduced and checked 12-sector firm types (according to ISIC/NACE 12) on generator create_firms()
29. Production logic: buying inputs from sectors, according to technical matrix coefficients at the same proportion of labor input
30. Households buy from all sectors, following final demand IBGE's table
31. Government budget division to participate in the intermediate market included.
32. Government labor market distinctions (fixed exogenous values by year) were implemented
33. Government firms participate in the consumption market, 
additionally from being responsible for infrastructure and policy.
34. Intermediate market for sectoral firms
35. Regionalization of metropolis versus rest of Brazil via decomposition of input-output official matrix
36. Regionalization of final demands
37. Implementation of new official urban concentration areas from IBGE, updated in 2022--CONURBs. 
Due to numerous mentions in the code, we will continue to use **ACPs despite them having been updated to CONURBs.**
38. Percentage of firms per sectors are now read from the data (not parameters anymore)


### Published at JASSS

## https://www.jasss.org/25/1/8.html

Policymakers' role in decision-making on alternative policies is facing restricted budgets and an uncertain future. 
The need to decide on priorities and handle effects across policies has made their task even more difficult. 
For instance, housing policies involve heterogeneous characteristics of the properties themselves and the intricacy 
of housing markets within the spatial context of cities. Here, we have proposed PolicySpace2 (PS2) as an adapted and 
extended version of the open source PolicySpace agent-based model. PS2 is a computer simulation that relies 
on empirically detailed spatial data to model real estate, along with labor, credit, and goods and services markets. 
Interaction among workers, firms, a bank, households and municipalities follow the literature benchmarks by 
integrating economic, spatial and transport research. PS2 is applied here as a comparison of three competing public 
policies aimed at reducing inequality and alleviating poverty: (a) house acquisition by the government and 
distribution to lower income households, (b) rental vouchers and (c) monetary aid. Within the model context, 
monetary aid, that is smaller amounts of help for a larger number of households, improves the economy in terms 
of production, consumption, reduction of inequality and maintenance of financial duties. PS2 is also a framework 
that can be further adapted to a number of related research questions.  


This is an evolution of PolicySpace  
<img alt="GitHub stars" src="https://img.shields.io/github/stars/bafurtado/policyspace.svg?color=orange">  ![GitHub forks](https://img.shields.io/github/forks/bafurtado/policyspace)
 ---

Available here: https://github.com/BAFurtado/PolicySpace and published as a book 
here: https://www.researchgate.net/publication/324991027_PolicySpace_agent-based_modeling

 
 **FURTADO, Bernardo Alves. PolicySpace: agent-based modeling. IPEA: Brasília, 2018.** 

This was an open agent-based model (ABM) with three markets and a tax scheme that empirically simulates 46 Brazilian
metropolitan regions. Now, we have also **added a credit market, a housing construction scheme and an incipient 
land market mechanism**. 

