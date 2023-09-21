# New model. 
## Enhancements and Improvements in PolicySpace2

## Todo
1. Fix labels plots (name instead of code)

### Changes made in this version
1. Addresses chosen with spatial join (instead of individually). Enhancement: faster run
2. Fixed bug construction firm planning house
3. Introduced new parameter to control for availability of lot space supply for construction
4. Changed output figures to include lower-upper bound confidence intervals, instead of plotting all runs lines.
5. Sales and rental markets now use submarkets in quartiles by income distributions. 
   Families in the lower income quartile search for houses in the lower quartile of quality
6. Endogenized decision to enter house market: based on employability, renting status, funds available, level consumption
7. Ordered, organized decision on consumption
8. Changed price and production decisions based on Dawid, 2018 benchmarks
9. Checked bank loan systems. 
10. Included new graph reports: wages received, wages paid, firms' stocks, population
11. Sorting of firms in the labor market made by per capita revenue!
12. Introduced parameters PRICE_RUGGEDNESS so that prices reduction is more rugged than when increasing
13. Now there is endogenous qualification growth, observing dropout average magnitude
14. Plotting with lower upper bounds (2std) for sensitivity comparison
15. Introduced test of collecting consumption taxes at consumers' municipality rather than firms'
16. Migrated from using `OGR from OSGEO` to regular `geopandas.DataFrames`
17. Initial real estate average area estimates of empirical values from FIPEZAP in the beginning are read from file, when available

------
``` python 3.11```
------
------
Previous work below
------
PolicySpace2: Real Estate Modeling
------
<img alt="GitHub stars" src="https://img.shields.io/github/stars/bafurtado/policyspace2.svg?color=orange">  ![GitHub All Releases](https://img.shields.io/github/downloads/bafurtado/policyspace2/total) ![GitHub](https://img.shields.io/github/license/bafurtado/policyspace2)  ![GitHub forks](https://img.shields.io/github/forks/bafurtado/policyspace2)
------

<a href="https://www.comses.net/codebases/c8775158-4360-46d8-bac8-be94502b04b0/releases/1.2.0/"><img src="https://www.comses.net/static/images/icons/open-code-badge.png" align="left" height="64" width="64" ></a>

---
#### Reviewed code at ComSES

## Post-PolicySpace2 work

1. Optimal policy, which, where, and why: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4276602
2. Machine Learning Simulates Agent-Based Model Towards Policy: https://arxiv.org/abs/2203.02576

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

### Collaborators
Bernardo Alves Furtado -- https://sites.google.com/view/bernardo-alves-furtado

Francis Tseng --  http://frnsys.com

![GitHub labels](https://img.shields.io/github/labels/atom/atom/help-wanted)

### Funding

Developed by Bernardo Alves Furtado, funded primarily by Institute of Applied Economic Research (IPEA) 
[www.ipea.gov.br] with one-period grant from [https://www.cepal.org/pt-br/sedes-e-escritorios/cepal-brasilia] 
(CEPAL-Brasília) and International Policy Centre (https://ipcig.org/). 
BAF acknowledges receiving a grant of productivity by National Council of Research (CNPq) [www.cnpq.br].

#### How do I get set up?

We recommend using conda  and creating an environment that includes all libraries simultaneously.

You do not need to inform the python version. Let the configuration of current libraries requirements decide.

The line below set on `python 3.11` on August 18, 2023

`conda create -n ps3 -c conda-forge fiona geopandas shapely gdal pandas numba descartes scipy seaborn pyproj matplotlib 
six cycler statsmodels joblib scikit-learn flask flask-wtf psutil`

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

Example:

```
python main.py -c 2 -n 10 run
```

#### Running tests

A few tests are implemented in `tests.py`. You may run them as: 
```angular2html
python tests.py
```

#### Sensitivity analysis

Runs simulation over a range of values for a specific parameter. For continuous parameters, the syntax is
`NAME:MIN:MAX:NUMBER_STEPS`. For boolean parameters, just provide the parameter name.
It now also accepts selected "PROCESSING_ACPS-BRASILIA-CAMPINAS-FORTALEZA-BELO HORIZONTE"

Example:

```
python main.py sensitivity ALPHA:0:1:7
```

Will run the simulation once for each value `ALPHA=0`, `ALPHA=0.17`, `ALPHA=0.33`, ... `ALPHA=1`.

You can also set up multiple sensitivity runs at once.

For example:

```
python main.py sensitivity MARKUP:.05:.15:7 PRODUCTIVITY_EXPONENT:.4:.6:3
```

is equivalent to running the previous two examples in sequence.

For multiple combinations of parameters one may try the following rules

Include first the params, separated by '+', then '*' and then the list of values also '+'
Such as 'param1+param2*1+2*10+20'.
Thus,  producing the dict: {'param1': ['10', '20'], 'param2': ['10', '20']}
```
python main.py sensitivity PRODUCTIVITY_EXPONENT+PRODUCTIVITY_MAGNITUDE_DIVISOR*.3+.4*10+20
```
#### Distributions

Runs simulation over a different distribution combinations: `ALTERNATIVE0: True/False, FPM_DISTRIBUTION: True/False`.

Example:

```
python main.py -n 2 -c 2 distributions
```

#### ACPs

Runs simulation over a different ACPs.

Example:

```
python main.py -n 2 -c 2 acps
```

#### Regenerating plots

You can regenerate plots for a set of runs by using:

```
python main.py make_plots /path/to/output
```

In Windows, make sure to use double quotes " " and backward slashes as in:

```
python main.py make_plots
"..\run__2017-11-01T11_59_59.240250_bh"
```

### Running the web interface

There is a preliminary web interface in development. 

I have not checked this interface in quite some time.

To run the server:

```
python main.py web

Then open `localhost:5000` in your browser.
