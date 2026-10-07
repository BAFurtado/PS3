"""
Calibration by history matching (LHS waves) or Sobol screening: sample → score → plausible-box / sensitivity

CLI:
    # Run one wave: Latin hypercube over CALIBRATION_PARAMETERS and CALIBRATION_OPTIONS, every set in every region
    # of calibration_regions (default design, see calibration_conf.py)
    python -m analysis.calibration.sample run-sample --samples 32 --cpus 10

    # Score a completed (or partially completed) results folder
    python -m analysis.calibration.sample score path/to/calibration_dir/

    # Parameter box of the sets not ruled out, to paste into calibration_conf.py for the next wave
    python -m analysis.calibration.sample plausible-box path/to/calibration_dir/

    # Sobol design and Total-Order Sobol Indices (needs SALib and N >= 512 to be stable)
    python -m analysis.calibration.sample run-sample --design sobol --samples 512 --cpus 4
    python -m analysis.calibration.sample sensitivity path/to/calibration_dir/

Interruption recovery:
    run-sample writes jobs.json to the output directory before dispatching.
    Each completed run writes a DONE sentinel file. If execution is interrupted,
    run resume to clean partial outputs and continue from where it left off.
"""
import os
import sys
import copy
import json
import logging
import datetime
from collections import defaultdict
from glob import glob
from datetime import datetime

import click
import numpy as np
import pandas as pd
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
from scipy.stats import qmc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import analysis.calibration.calibration_conf as calibration_conf
import analysis.calibration.levels as levels
import conf
from analysis.output import columns_for
from checkpoint import save_jobs, pending_jobs
from simulation import Simulation
from main import gen_output_dir

logger = logging.getLogger('main')


# ── FITNESS ───────────────────────────────────────────────────────────────────

_OBSERVED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _period_to_date(period, freq: str):
    """Convert a fetch_municipal_validation_data.py period code to a date.

    'month' periods are YYYYMM. 'quarter' periods are YYYY0Q (Q = quarter
    number 1-4, per SIDRA's D3C convention), converted to that quarter's
    first month. 'year' periods are plain YYYY.
    """
    period = str(period)
    year = int(period[:4])
    if freq == "month":
        month = int(period[4:6])
    elif freq == "quarter":
        month = (int(period[4:6]) - 1) * 3 + 1
    elif freq == "year":
        month = 1
    else:
        raise ValueError(freq)
    return datetime(year, month, 1).date()


def _load_observed_moments(start: str, end: str) -> dict:
    """
    Empirical mean/std for each fitness target, over [start, end], from the BH data fetched by
    fetch_municipal_validation_data.py. GDP growth is deflated by the calendar-year IPCA-BH: PIB dos
    Municípios is nominal and the model runs in real terms (it has no nominal anchor). Gini is observed only
    for the state (PNADC per-capita household income, mg_gini.csv) and is returned when that file exists.
    """
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    def _read(fname, value_col, freq):
        df = pd.read_csv(os.path.join(_OBSERVED_DATA_DIR, fname))
        df["date"] = df["period"].apply(lambda p: _period_to_date(p, freq))
        return df[["date", value_col]]

    def _windowed(df):
        return df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    inflation = _read("bh_inflation.csv", "inflation", "month")
    unemployment = _windowed(_read("bh_unemployment.csv", "unemployment", "quarter"))["unemployment"]
    nominal_growth = _windowed(_read("bh_gdp_growth.csv", "gdp_growth", "year")).set_index("date")["gdp_growth"]

    annual_ipca = (1 + inflation["inflation"]).groupby(inflation["date"].map(lambda d: d.year)).prod() - 1
    years = nominal_growth.index.map(lambda d: d.year)
    covered = years.isin(annual_ipca.index)
    real_growth = (1 + nominal_growth[covered].values) / (1 + annual_ipca.loc[years[covered]].values) - 1
    inflation = _windowed(inflation)["inflation"]

    observed = {
        "gdp_growth_mean":   float(np.mean(real_growth)),
        "gdp_growth_std":    float(np.std(real_growth, ddof=1)),
        "unemployment_mean": float(unemployment.mean()),
        "unemployment_std":  float(unemployment.std()),
        "inflation_mean":    float(inflation.mean()),
        "inflation_std":     float(inflation.std()),
    }
    gini_path = os.path.join(_OBSERVED_DATA_DIR, "mg_gini.csv")
    if os.path.exists(gini_path):
        gini = _windowed(_read("mg_gini.csv", "gini", "year"))["gini"]
        observed["gini_mean"] = float(gini.mean())
        observed["gini_std"] = float(gini.std())
    return observed


def simulated_moments(sim_df: pd.DataFrame) -> dict | None:
    """
    Simulated counterparts of the observed moments over [burn_in_end, target_end_year], or None when the run
    is unusable. GDP growth is annual, from calendar-year real GDP (gdp_level over price_level), built like the
    observed series; the base year before burn_in_end is used only as the first level.
    """
    settings = calibration_conf.CALIBRATION_SETTINGS
    required = {"month", "gdp_level", "price_level", "unemployment", "gini_index", "inflation"}
    if not required.issubset(sim_df.columns):
        return None

    start, end = settings["burn_in_end"], settings["target_end_year"]
    month = pd.to_datetime(sim_df["month"])
    df = sim_df[(month >= start) & (month < end)]
    if df.empty:
        return None

    real_gdp = (sim_df["gdp_level"] / sim_df["price_level"]).groupby(month.dt.year).sum()
    first, last = pd.Timestamp(start).year, pd.Timestamp(end).year - 1
    # only whole years count, so a run that stopped mid-year does not score a partial year as a slump
    months_in_year = month.dt.year.value_counts()
    whole = [y for y in real_gdp.index if months_in_year[y] == 12 and first - 1 <= y <= last]
    growth = real_gdp.loc[whole].pct_change().dropna()
    if len(growth) < 2:
        return None

    return {
        "gdp_growth_mean":   float(growth.mean()),
        "gdp_growth_std":    float(growth.std()),
        "unemployment_mean": float(df["unemployment"].mean()),
        "unemployment_std":  float(df["unemployment"].std()),
        "inflation_mean":    float(df["inflation"].mean()),
        "inflation_std":     float(df["inflation"].std()),
        "gini_mean":         float(df["gini_index"].mean()),
        "gini_std":          float(df["gini_index"].std()),
    }


def fitness_from_moments(simulated: dict, observed: dict, weights: dict) -> float:
    """Weighted mean relative deviation, sum_m w_m |sim_m - obs_m| / |obs_m|, over the moments in `weights`."""
    return sum(w * abs(simulated[m] - observed[m]) / abs(observed[m]) for m, w in weights.items())


def equal_weights() -> dict:
    moments = calibration_conf.CALIBRATION_SETTINGS["fitness_moments"]
    return {m: 1 / len(moments) for m in moments}


def calculate_fitness(sim_df: pd.DataFrame, weights: dict | None = None) -> float:
    """
    Fitness of one run: fitness_from_moments with `weights` (equal over fitness_moments by default).
    Returns 999.0 on failure.
    """
    settings = calibration_conf.CALIBRATION_SETTINGS
    simulated = simulated_moments(sim_df)
    if simulated is None:
        return 999.0
    observed = _load_observed_moments(settings["burn_in_end"], settings["target_end_year"])
    return fitness_from_moments(simulated, observed, weights or equal_weights())


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
@click.pass_context
def main(ctx):
    ctx.ensure_object(dict)


@main.command()
@click.option("-s", "--samples", default=None, type=int,
              help="LHS: number of sets; Sobol: N (power of 2). Overrides calibration_conf value.")
@click.option("-d", "--design", default=None, type=click.Choice(["lhs", "sobol"]),
              help="Sampling design. Overrides calibration_conf value.")
@click.option("-c", "--cpus", default=-1, help="Cores (-1 for all).")
@click.pass_context
def run_sample(ctx, samples, design, cpus):
    """Generate an LHS or Sobol sample and run simulations."""
    params_to_calibrate = calibration_conf.CALIBRATION_PARAMETERS
    options             = getattr(calibration_conf, "CALIBRATION_OPTIONS", {})
    settings            = calibration_conf.CALIBRATION_SETTINGS

    names = list(params_to_calibrate.keys())
    lbs   = [v[0] for v in params_to_calibrate.values()]
    ubs   = [v[1] for v in params_to_calibrate.values()]
    option_names = list(options.keys())

    n_samples = samples if samples is not None else settings["samples"]
    n_runs    = settings["runs_per_sample"]
    design    = design or settings.get("design", "sobol")

    problem = {
        "num_vars": len(names),
        "names":    names,
        "bounds":   list(zip(lbs, ubs))
    }
    choices = [{} for _ in range(n_samples)]
    if design == "lhs":
        unit = qmc.LatinHypercube(d=len(names) + len(option_names), seed=settings["lhs_seed"]).random(n_samples)
        scaled_samples = qmc.scale(unit[:, :len(names)], lbs, ubs)
        for j, name in enumerate(option_names):
            values = options[name]
            for i in range(n_samples):
                choices[i][name] = values[min(int(unit[i, len(names) + j] * len(values)), len(values) - 1)]
    else:
        if option_names:
            raise click.ClickException("The Sobol design takes no CALIBRATION_OPTIONS; fix them in params.py.")
        from SALib.sample import sobol as sobol_sampler
        if (n_samples & (n_samples - 1)) != 0:
            logger.warning(f"samples={n_samples} is not a power of 2.")
        scaled_samples = sobol_sampler.sample(
            problem,
            N=n_samples,
            calc_second_order=settings["sobol_calc_second_order"],
            seed=settings["sobol_seed"],
        )
    start_date = datetime.strptime(settings['target_start_year'], '%Y-%m-%d').date()
    end_date = datetime.strptime(settings['target_end_year'], '%Y-%m-%d').date()
    period = {"STARTING_DAY": start_date, "TOTAL_DAYS": (end_date - start_date).days}
    confs = [{**dict(zip(names, (float(x) for x in scaled_samples[i]))), **(choices[i] if i < len(choices) else {}),
              **period} for i in range(len(scaled_samples))]

    output_dir = gen_output_dir("calibration")

    _save_meta(output_dir, problem, n_samples, n_runs, settings, design, options)
    multiple_runs(confs, n_runs, cpus, output_dir, settings["calibration_regions"])

    logger.info(f"Done. Results in: {output_dir}")


@main.command()
@click.argument("root_dir")
def score(root_dir):
    """Score an existing results folder."""
    score_calibration(root_dir)


@main.command("plausible-box")
@click.argument("root_dir")
@click.option("--cutoff", default=None, type=float, help="Implausibility cutoff. Overrides calibration_conf value.")
def plausible_box_cmd(root_dir, cutoff):
    """Print the parameter box of the sets not ruled out, for the next wave."""
    plausible_box(root_dir, cutoff)


@main.command()
@click.argument("root_dir")
def sensitivity(root_dir):
    """Compute Total-Order Sobol Indices (S_Ti) from scored results."""
    compute_sensitivity(root_dir)


@main.command()
@click.argument("root_dir")
@click.option("-c", "--cpus", default=-1, help="Cores (-1 for all).")
def resume(root_dir, cpus):
    """Resume an interrupted calibration run from root_dir/jobs.json."""
    try:
        pending, cleaned = pending_jobs(root_dir)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    if not pending:
        logger.info("All jobs already completed — nothing to resume.")
        return

    logger.info(f"Cleaned {cleaned} partial run(s). Resuming {len(pending)} job(s)...")
    _dispatch(pending, cpus, desc="Resuming")


# ── EXECUTION ─────────────────────────────────────────────────────────────────

def multiple_runs(overrides: list, runs: int, cpus: int, output_dir: str, regions: list):
    """Dispatch all (parameter set × region × Monte Carlo run) jobs in parallel, to <set>/<REGION>/<run>."""
    # Common random numbers: replication i uses the same seed in every set, so sets differ by their parameters,
    # not their draws, and every run is reproducible from its conf.json
    seed_base = calibration_conf.CALIBRATION_SETTINGS.get("seed_base", 1000)
    job_specs = []
    for n, o in enumerate(overrides):
        for region in regions:
            p = copy.deepcopy(conf.PARAMS)
            p.update(o)
            p["PROCESSING_ACPS"] = [region]
            for i in range(runs):
                job_specs.append({"path": os.path.join(output_dir, str(n), region.replace(" ", "_"), str(i)),
                                  "params": {**p, "SEED": seed_base + i}})
    save_jobs(output_dir, job_specs, cpus)
    build_populations(job_specs, output_dir)

    _dispatch(job_specs, cpus, desc="Calibration runs")
    logger.info("All runs completed.")


def build_populations(job_specs: list, output_dir: str):
    """Creates, one region at a time, the population file each region's runs load (StoragedAgents), so parallel
    runs never write it at the same time"""
    done = set()
    for job in job_specs:
        region = tuple(job["params"]["PROCESSING_ACPS"])
        if region in done:
            continue
        done.add(region)
        path = os.path.join(output_dir, "populations", "_".join(region).replace(" ", "_"))
        os.makedirs(path, exist_ok=True)
        sim = Simulation(copy.deepcopy(job["params"]), path)
        if not os.path.isfile("{}.agents".format(sim.output.save_name)):
            logger.info(f"Creating the population of {region[0]}")
            sim.generate()


def single_run(params: dict, path: str):
    """Run one simulation and write outputs to path."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "conf.json"), "w") as f:
        json.dump({"PARAMS": params}, f, indent=4, default=str)

    conf.RUN["SAVE_PLOTS_FIGURES"] = False
    sim = Simulation(params, path)
    sim.initialize()
    sim.run(log=False)
    open(os.path.join(path, "DONE"), "w").close()


# ── SCORING ───────────────────────────────────────────────────────────────────

def score_calibration(root_dir: str) -> pd.DataFrame:
    """Score each parameter set on the targets of calibration_conf ('levels' or 'series')"""
    if calibration_conf.CALIBRATION_SETTINGS.get("targets", "series") == "levels":
        return score_levels(root_dir)
    return score_series(root_dir)


def _set_dirs(root_dir: str) -> list:
    return sorted([d for d in glob(os.path.join(root_dir, "*/")) if os.path.basename(os.path.normpath(d)).isdigit()],
                  key=lambda d: int(os.path.basename(os.path.normpath(d))))


def _runs(ps_dir: str):
    """(region, run dir) of a set: <set>/<REGION>/<run>, or <set>/<run> for a batch of one region"""
    for rd in sorted(glob(os.path.join(ps_dir, "*/"))):
        name = os.path.basename(os.path.normpath(rd))
        if name.isdigit():
            yield None, rd
        elif name != "populations":
            for sub in sorted(glob(os.path.join(rd, "*/"))):
                if os.path.basename(os.path.normpath(sub)).isdigit():
                    yield name.replace("_", " "), sub


def _tracked() -> list:
    return list(calibration_conf.CALIBRATION_PARAMETERS) + list(getattr(calibration_conf, "CALIBRATION_OPTIONS", {}))


def _read_stats(rd: str):
    csv_path = os.path.join(rd, "stats.csv")
    if not os.path.exists(csv_path) or not os.path.exists(os.path.join(rd, "DONE")):
        return None
    df = pd.read_csv(csv_path, header=None, sep=";")
    df.columns = columns_for("stats", df.shape[1])
    return df


def _snapshot(rd: str) -> dict:
    conf_file = os.path.join(rd, "conf.json")
    if not os.path.exists(conf_file):
        return {}
    with open(conf_file) as f:
        data = json.load(f)
    return {k: data["PARAMS"].get(k) for k in _tracked()}


def score_levels(root_dir: str) -> pd.DataFrame:
    """
    Score each parameter set on the levels of data/level_targets.csv in every region and write calibration_scores.csv
    (one row per set) and calibration_levels.csv (one row per set and region).

    A set is ruled out (max_implausibility = inf) when any of its runs fails a hard constraint (levels.explodes) or a
    region has no completed run. Otherwise each level is the mean over the region's seeds, and its implausibility is
    its distance outside the band over sqrt(seed_sd^2 + (model_discrepancy x band midpoint)^2), seed_sd the region's
    seed sd of that level pooled over sets with >= 2 seeds. max_implausibility is the largest over regions and levels,
    score the mean.
    """
    settings = calibration_conf.CALIBRATION_SETTINGS
    regions = settings["calibration_regions"]
    window = settings.get("levels_window", 36)
    targets = levels.load_targets(regions)
    moments = list(next(iter(targets.values())))

    sets = []
    for ps_dir in _set_dirs(root_dir):
        by_region, snapshot, exploded = defaultdict(list), {}, False
        for region, rd in _runs(ps_dir):
            snapshot = snapshot or _snapshot(rd)
            df = _read_stats(rd)
            if df is None or len(df) <= 24 + window:
                continue
            m = levels.level_moments(df, window)
            exploded |= levels.explodes(m)
            by_region[region].append(m)
        sets.append((ps_dir, snapshot, {r: pd.DataFrame(v) for r, v in by_region.items()}, exploded))
    if not sets:
        raise click.ClickException(f"No scorable runs under {root_dir}.")

    seed_sd = {}
    for region in regions:
        multi = [runs[region][moments] for _, _, runs, _ in sets if region in runs and len(runs[region]) >= 2]
        seed_sd[region] = (pd.concat([r.var(ddof=1) for r in multi], axis=1).mean(axis=1) ** 0.5
                           if multi else pd.Series(0.0, index=moments)).fillna(0.0)

    results, rows = [], []
    for ps_dir, snapshot, runs, exploded in sets:
        implaus = []
        for region in regions:
            if region not in runs:
                implaus.append(np.inf)
                continue
            mean = runs[region][moments].mean()
            row = {"set": os.path.basename(os.path.normpath(ps_dir)), "region": region, "n_runs": len(runs[region])}
            for k in moments:
                i = levels.implausibility(mean[k], targets[region][k], seed_sd[region][k],
                                          settings["model_discrepancy"])
                row[k], row[f"I_{k}"] = mean[k], i
                implaus.append(i)
            rows.append(row)
        complete = all(r in runs for r in regions)
        results.append({
            **snapshot,
            "score":              np.mean(implaus) if complete and not exploded else np.inf,
            "max_implausibility": max(implaus) if complete and not exploded else np.inf,
            "exploded":           exploded,
            "regions":            len(runs),
            "path":               ps_dir,
        })

    score_df = pd.DataFrame(results).sort_values("score").reset_index(drop=True)
    output_path = os.path.join(root_dir, "calibration_scores.csv")
    score_df.to_csv(output_path, index=False)
    pd.DataFrame(rows).to_csv(os.path.join(root_dir, "calibration_levels.csv"), index=False)
    pd.DataFrame(seed_sd).to_csv(os.path.join(root_dir, "seed_sd.csv"))

    logger.info(f"Scores saved to: {output_path}")
    print(score_df[_tracked() + ["score", "max_implausibility", "exploded"]].head(10).to_string(index=False))
    return score_df


def score_series(root_dir: str) -> pd.DataFrame:
    """
    Score each parameter set and write calibration_scores.csv (plus score_weights.csv) to root_dir.

    Each set is scored on its moments averaged across seeds (|E[sim] - obs|, not E|sim - obs|, so seed noise
    does not inflate the distance). Weights follow fitness_weights: "inverse_noise" weights each relative
    deviation by |obs| / sqrt(seed_sd^2 + (model_discrepancy * obs)^2), seed_sd pooled across sets with >= 2
    seeds. Seed noise alone would hand nearly all the weight to the most stable moment (unemployment); with the
    discrepancy term the weights stay near equal and only down-weight moments that are noisy across seeds.
    max_implausibility is the largest, over moments, of
    |E[sim] - obs| / sqrt(seed_sd^2 + (model_discrepancy * obs)^2); `plausible-box` filters on it.
    """
    settings = calibration_conf.CALIBRATION_SETTINGS
    tracked = _tracked()
    moments = settings["fitness_moments"]
    observed = _load_observed_moments(settings["burn_in_end"], settings["target_end_year"])

    sets = []
    for ps_dir in _set_dirs(root_dir):
        runs, param_snapshot = [], {}
        for _, rd in _runs(ps_dir):
            param_snapshot = param_snapshot or _snapshot(rd)
            df = _read_stats(rd)
            if df is not None:
                m = simulated_moments(df)
                if m is not None:
                    runs.append(m)
        if runs:
            sets.append((ps_dir, param_snapshot, pd.DataFrame(runs)))

    if not sets:
        raise click.ClickException(f"No scorable runs under {root_dir}.")

    # pooled within-set (seed) sd of each moment, from sets with >= 2 seeds
    multi = [r for _, _, r in sets if len(r) >= 2]
    seed_sd = (pd.concat([r[moments].var(ddof=1) for r in multi], axis=1).mean(axis=1) ** 0.5
               if multi else pd.Series(np.nan, index=moments))
    # tolerance of each moment: seed noise combined with the model discrepancy, as in the implausibility
    obs = pd.Series(observed)[moments]
    scale = np.sqrt(seed_sd.fillna(0) ** 2 + (settings["model_discrepancy"] * obs) ** 2)
    if settings["fitness_weights"] == "inverse_noise":
        if not multi:
            logger.warning("inverse_noise weights need >= 2 seeds per set; using the discrepancy term alone.")
        raw = obs.abs() / scale
        weights = (raw / raw.sum()).to_dict()
    else:
        weights = equal_weights()

    results = []
    for ps_dir, param_snapshot, runs in sets:
        mean = runs[moments].mean()
        implaus = (mean - pd.Series(observed)[moments]).abs() / scale
        run_scores = [fitness_from_moments(r, observed, weights) for r in runs.to_dict("records")]
        results.append({
            **param_snapshot,
            "score":              fitness_from_moments(mean, observed, weights),
            "score_std":          np.std(run_scores),
            "max_implausibility": implaus.max(),
            **{f"{m}_sim": mean[m] for m in moments},
            **{f"I_{m}": implaus[m] for m in moments},
            "n_runs":             len(runs),
            "path":               ps_dir,
        })

    score_df    = pd.DataFrame(results).sort_values("score").reset_index(drop=True)
    output_path = os.path.join(root_dir, "calibration_scores.csv")
    score_df.to_csv(output_path, index=False)
    pd.DataFrame({"observed": pd.Series(observed)[moments], "seed_sd": seed_sd,
                  "weight": pd.Series(weights)}).to_csv(os.path.join(root_dir, "score_weights.csv"))

    logger.info(f"Scores saved to: {output_path}")
    print("weights:", {m: round(w, 3) for m, w in weights.items()})
    print(score_df[tracked + ["score", "score_std", "max_implausibility"]].head(10).to_string(index=False))

    return score_df


def plausible_box(root_dir: str, cutoff: float | None = None) -> dict:
    """
    Parameter ranges spanned by the sets that are not ruled out (max_implausibility <= cutoff), to paste into
    CALIBRATION_PARAMETERS for the next wave. This is the box around the surviving sets, so it may still
    contain implausible corners; the next wave narrows it again.
    """
    settings = calibration_conf.CALIBRATION_SETTINGS
    cutoff = settings["implausibility_cutoff"] if cutoff is None else cutoff
    tracked = list(calibration_conf.CALIBRATION_PARAMETERS.keys())
    options = getattr(calibration_conf, "CALIBRATION_OPTIONS", {})
    scores_path = os.path.join(root_dir, "calibration_scores.csv")
    if not os.path.exists(scores_path):
        raise click.ClickException(f"calibration_scores.csv not found in {root_dir}. Run 'score' first.")
    df = pd.read_csv(scores_path)
    keep = df[df["max_implausibility"] <= cutoff]
    print(f"{len(keep)} of {len(df)} sets not ruled out at implausibility <= {cutoff}")
    if keep.empty:
        print("None survive: the observed moments are out of reach in this box, or the discrepancy is too tight.")
        levels_path = os.path.join(root_dir, "calibration_levels.csv")
        if settings.get("targets", "series") == "levels" and os.path.exists(levels_path):
            lv = pd.read_csv(levels_path)
            print("Exploded sets:", int(df["exploded"].sum()), "of", len(df))
            print("Most binding level per set and region (count):")
            print(lv[[c for c in lv.columns if c.startswith("I_")]].idxmax(axis=1).value_counts().to_string())
        else:
            print("Most binding moment per set (count):")
            print(df[[f"I_{m}" for m in settings["fitness_moments"]]].idxmax(axis=1).value_counts().to_string())
        return {}
    for name, values in options.items():
        print(f'    "{name}": {sorted(keep[name].dropna().unique().tolist())},   # was {values}')
    box = {p: [round(float(keep[p].min()), 4), round(float(keep[p].max()), 4)] for p in tracked}
    for p, (lo, hi) in box.items():
        old_lo, old_hi = calibration_conf.CALIBRATION_PARAMETERS[p]
        print(f'    "{p}": [{lo}, {hi}],   # was [{old_lo}, {old_hi}]')
    return box


# ── SENSITIVITY ───────────────────────────────────────────────────────────────

def compute_sensitivity(root_dir: str) -> pd.DataFrame:
    """
    Compute S_Ti from meta.json + calibration_scores.csv.
    Writes sobol_indices.csv and prints KEEP / FREEZE decisions.
    """
    settings = calibration_conf.CALIBRATION_SETTINGS

    meta_path = os.path.join(root_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {root_dir}.")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("design", "sobol") != "sobol":
        raise click.ClickException("Sobol indices need a Sobol (Saltelli) design; this batch is "
                                   f"{meta['design']}. Use 'score' and 'plausible-box' instead.")
    from SALib.analyze import sobol as sobol_analyze

    scores_path = os.path.join(root_dir, "calibration_scores.csv")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"calibration_scores.csv not found in {root_dir}. Run 'score' first.")

    problem   = meta["problem"]
    n_samples = meta["n_samples"]
    _scores   = pd.read_csv(scores_path)
    _scores["_idx"] = _scores["path"].apply(lambda p: int(os.path.basename(os.path.normpath(p))))
    Y         = _scores.sort_values("_idx")["score"].values

    expected = n_samples * (problem["num_vars"] + 2)
    if len(Y) != expected:
        raise ValueError(f"Y length {len(Y)} != expected {expected} (N*(k+2)).")

    si = sobol_analyze.analyze(
        problem, Y,
        calc_second_order=settings["sobol_calc_second_order"],
        print_to_console=False,
    )

    threshold = settings["freeze_threshold_sti"]
    sti_df = (
        pd.DataFrame({
            "parameter": problem["names"],
            "S_Ti":      si["ST"],
            "S_Ti_conf": si["ST_conf"],
        })
        .sort_values("S_Ti", ascending=False)
        .reset_index(drop=True)
    )
    sti_df["decision"] = np.where(sti_df["S_Ti"] < threshold, "FREEZE", "KEEP")

    output_path = os.path.join(root_dir, "sobol_indices.csv")
    sti_df.to_csv(output_path, index=False)

    print(f"\n--- S_Ti (freeze threshold: {threshold}) ---")
    print(sti_df.to_string(index=False))
    print(f"\nKEEP:   {sti_df[sti_df.decision=='KEEP']['parameter'].tolist()}")
    print(f"FREEZE: {sti_df[sti_df.decision=='FREEZE']['parameter'].tolist()}")

    return sti_df


# ── HELPERS ───────────────────────────────────────────────────────────────────

MAX_JOB_ATTEMPTS = 3


def _dispatch(job_specs: list, cpus: int, desc: str):
    """Run job specs in parallel processes, resubmitting any missing a DONE
    sentinel after a pass, up to MAX_JOB_ATTEMPTS each."""
    max_workers = None if cpus is None or cpus < 1 else cpus
    attempts = defaultdict(int)
    remaining = list(job_specs)
    bar = tqdm.tqdm(total=len(job_specs), desc=desc, unit="sim", dynamic_ncols=True)
    try:
        while remaining:
            for job in remaining:
                attempts[job["path"]] += 1
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(single_run, j["params"], j["path"]): j
                               for j in remaining}
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error("Run failed (%s): %s", futures[future]["path"], e)
            except BrokenExecutor as e:
                logger.warning("Worker pool broke: %s", e)

            finished = [j for j in remaining if os.path.exists(os.path.join(j["path"], "DONE"))]
            bar.update(len(finished))
            remaining = [j for j in remaining if j not in finished]

            abandoned = [j for j in remaining if attempts[j["path"]] >= MAX_JOB_ATTEMPTS]
            for job in abandoned:
                logger.error("Abandoning %s after %d attempt(s) without DONE.",
                             job["path"], attempts[job["path"]])
            remaining = [j for j in remaining if j not in abandoned]
            if remaining:
                logger.info("%d job(s) missing DONE after pass; retrying...", len(remaining))
    finally:
        bar.close()


def _save_meta(output_dir: str, problem: dict, n_samples: int,
               n_runs: int, settings: dict, design: str = "sobol", options: dict | None = None):
    """Write meta.json required by compute_sensitivity."""
    meta = {
        "problem":   problem,
        "options":   options or {},
        "design":    design,
        "n_samples": n_samples,
        "n_runs":    n_runs,
        "timestamp": datetime.now().isoformat(),
        "settings":  {k: v for k, v in settings.items() if k != "fitness_weights"},
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    main()