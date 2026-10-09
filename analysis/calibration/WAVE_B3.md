# Calibration wave B3: what to run on the server

History-matching wave of the frozen model (closure B: the city as a small open economy). One parameter set for every
city; the cities differ only by their data. The wave asks which parameter sets keep four capitals inside observed
bands for the levels below. It does not fit time paths.

Wave B2 (tag `b-freeze-2`) ran on a base model that has since changed: the trade base now counts the own-account
pools' part of each market (TRADE_BASE_OUTPUT 'market'), workers separate and find jobs at the PME 2010 monthly rates
(LABOUR_FLOWS 'data'), and the own-account pools hold the Census 2010 own-account share and produce
(OWN_ACCOUNT_POOL 'census'). B3 runs the box again on this base. Two dimensions of B2 are fixed at their defaults:
HOUSING_FINANCIAL_WEIGHT (no level moved with it in B2) and CONSTRUCTION_PLAN 'pipeline' (every B2 set that survived
used it).

## Code version

Run the tag **`b-freeze-3`** (branch `enmu`), the commit that adds this file. Do not run `b-freeze`, `b-freeze-2`,
`density-paper-2026` or any older commit: their defaults are earlier models.

```
git fetch --tags
git checkout b-freeze-3
```

Before running, **empty or delete `conf\params.py`** on the server. It is gitignored and loaded on top of the
defaults, so any key left in it from earlier batches would silently change the base model of every set.
`conf\run.py` needs nothing for this wave.

## Command (Anaconda prompt, repository root)

```
python -m analysis.calibration.sample run-sample --cpus 20
```

`--cpus` is the number of parallel runs. Each run takes up to about 1.4 GB of RAM, so at 20 runs leave about
30 GB free. The script first builds each city's population once (StoragedAgents), then runs everything in
parallel. It writes `output\calibration__<timestamp>\`.

If the run is interrupted:

```
python -m analysis.calibration.sample resume output\calibration__<timestamp>
```

When all runs have finished:

```
python -m analysis.calibration.sample score output\calibration__<timestamp>
python -m analysis.calibration.sample plausible-box output\calibration__<timestamp>
```

## Design

Everything below is set in `analysis/calibration/calibration_conf.py`, so the command takes no other arguments.

| | |
|---|---|
| Design | Latin hypercube, 32 sets (`lhs_seed` 42) |
| Cities | BELO HORIZONTE, FORTALEZA, GOIANIA, PALMAS (every set runs in all four) |
| Seeds | 2 per set and city (1000, 1001, the same in every set) |
| Period | 2010-01 to 2019-12 (10 years), 1 % of the population |
| Runs | 32 x 4 x 2 = 256 |
| Levels scored on | Mean of the last 36 months (2017-2019) |

Run time on the desktop: Belo Horizonte about 30 min a run, Fortaleza 15, Goiânia 11, Palmas 4. That makes about
65 CPU hours for the wave, roughly 3.5 hours on 20 parallel runs.

### Parameters that vary

| Parameter | Range | Default |
|---|---|---|
| PRODUCTIVITY_EXPONENT | 0.5 to 0.8 | 0.65 |
| STICKY_PRICES | 0.3 to 0.9 | 0.7 |
| LABOR_MARKET | 0.4 to 0.9 | 0.8 |
| BUILD_VACANCY_SENSITIVITY | 7 to 19 | 13 |

Every other parameter stays at its value in `conf/default/params.py`, among them HOUSING_FINANCIAL_WEIGHT (60) and
CONSTRUCTION_PLAN ('pipeline'). PRODUCTIVITY_MAGNITUDE_DIVISOR is set at start-up from IBGE municipal value added, and
the labour flows come from PME 2010, so neither is calibrated.

### Targets (`analysis/calibration/data/level_targets.csv`)

| Level | Band | Source |
|---|---|---|
| Unemployment | per city: BH 0.069-0.155, FOR 0.080-0.126, GYN 0.054-0.097, PMW 0.064-0.138 | Census 2010 rate of the active (low), highest PNAD Contínua annual rate 2012-2019 (high) |
| Household income per resident, R$ of 2010 a month, deflated by the model price level | per city: BH 939-1392, FOR 637-945, GYN 896-1328, PMW 992-1471 | Census 2010 income per resident x the state's PNAD Contínua real growth 2012 to 2017-19 (low), times the national under-reporting factor (high); model income on the Census basis (wages before the worker's contributions and income tax, rent received) |
| Non-tradable inflation, %/yr (2011-2019) | 0 to 2.35 | IPCA non-tradables over tradables 2011-2015 |
| Vacancy | 0.08 to 0.13 | Density paper Table 2 |
| Consumption / GDP | 0.55 to 0.65 | Density paper Table 2 |
| Houses built per 1,000 residents a year | 2 to 4 | Density paper Table 2 |
| House price / family income (years) | 3.7 to 6.3 | Census 2010, 25 capitals |
| House price / median worker's monthly wage | 81 to 119 | Census 2010, 25 capitals |
| Housing stock / annual family income | 3.1 to 4.7 | Census 2010, 25 capitals |

Hard constraints: a set is ruled out if **any** of its runs breaks one of these.

- Unemployment of 30 % or more, either in the 36-month mean or in any month after month 24.
- Price growth of 5 %/yr or more.
- Real GDP growth outside -5 to +15 %/yr.
- Money created or lost outside the ledger.

Implausibility of a level is its distance outside the band divided by
`sqrt(seed_sd^2 + (0.10 x band midpoint)^2)`, where seed_sd is the spread across the two seeds. Inside the band it
is 0. A set survives when every level in every city is at 3 or less. `score` writes the following files:

- `calibration_scores.csv`: one row per set, with score = mean implausibility and max_implausibility.
- `calibration_levels.csv`: one row per set and city, with every level and its implausibility.
- `seed_sd.csv`.

## What to send back

Send the whole `output\calibration__<timestamp>\` folder, zipped. Leave out `populations\`. If the zip is too large,
send these instead:

- `calibration_scores.csv`, `calibration_levels.csv`, `seed_sd.csv`, `meta.json`, `jobs.json`.
- Every run's `stats.csv` and `conf.json`, under `<set>\<CITY>\<seed>\`.

## Next wave

`plausible-box` prints the ranges spanned by the sets that survive. Those become the ranges of the next wave in
`calibration_conf.py`. If no set survives, it prints the level that binds
most often. That level is a model question to settle before any further wave, not a reason to widen the box.
