"""
Tier 1 Sobol Scouting: sample → score → sensitivity

CLI:
    # Run full parameter space sampling (power of 2 recommended for --samples)
    python -m analysis.calibration.sample run-sample --samples 64 --cpus 4

    # Resume an interrupted run (cleans partial outputs, re-runs missing jobs)
    python -m analysis.calibration.sample resume path/to/calibration_dir/

    # Score a completed (or partially completed) results folder
    python -m analysis.calibration.sample score path/to/calibration_dir/

    # Compute Total-Order Sobol Indices from scored results
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
from glob import glob
from datetime import datetime

import click
import joblib
import numpy as np
import pandas as pd
import tqdm
from contextlib import contextmanager
from joblib import Parallel, delayed
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import analysis.calibration.calibration_conf as calibration_conf
import conf
from checkpoint import save_jobs, pending_jobs
from simulation import Simulation
from main import gen_output_dir
from analysis.output import OUTPUT_DATA_SPEC

logger = logging.getLogger('main')


# ── FITNESS ───────────────────────────────────────────────────────────────────

def calculate_fitness(sim_df: pd.DataFrame) -> float:
    """
    Weighted distance between simulated and observed BH moments (mean, std)
    over the burn-in window. Returns 999.0 on failure.
    """
    settings = calibration_conf.CALIBRATION_SETTINGS
    weights  = []#settings["fitness_weights"]

    # TODO: replace with values computed from actual observed series
    OBSERVED = {
        "gdp_growth_mean":   0.10,
        "gdp_growth_std":    0.018,
        "unemployment_mean": 0.090,
        "unemployment_std":  0.021,
        "gini_mean":         0.540,
        "gini_std":          0.018,
        "inflation_mean":    0.065/12,
        "inflation_std":     0.031/np.sqrt(12),
    }

    required = {"gdp_growth", "unemployment", "gini_index", "inflation"}
    if not required.issubset(sim_df.columns):
        return 999.0
    start, end = settings["target_start_year"],settings["target_end_year"]
    if "month" in sim_df.columns:
        df = sim_df[(sim_df["month"]>= start) & (sim_df["month"] <= end)]
    elif "year" in sim_df.columns:
        df = sim_df[(sim_df["year"] >= start) & (sim_df["year"] <= end)]
    else:
        df = sim_df

    if df.empty:
        return 999.0

    SIMULATED = {
        "gdp_growth_mean":   float((df["gdp_growth"].mean() + 1) ** 12 - 1),
        "gdp_growth_std":    float(df["gdp_growth"].std())*np.sqrt(12),
        "unemployment_mean": float(df["unemployment"].mean()),
        "unemployment_std":  float(df["unemployment"].std()),
        "gini_mean":         float(df["gini_index"].mean()),
        "gini_std":          float(df["gini_index"].std()),
        "inflation_mean":    float(df["inflation"].mean()),
        "inflation_std":     float(df["inflation"].std()),
    }

    moment_weights = {
        "gdp_growth_mean":  1 / 8,
        "gdp_growth_std":    1 / 8,
        "unemployment_mean": 1 / 8,
        "unemployment_std":  1 / 8,
        "gini_mean":         1/ 8,
        "gini_std":          1 / 8,
        "inflation_mean":    1 / 8,
        "inflation_std":     1 / 8,
    }

    return sum(
        moment_weights[m] * abs(SIMULATED[m] - OBSERVED[m])
        for m in OBSERVED
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
@click.pass_context
def main(ctx):
    ctx.ensure_object(dict)


@main.command()
@click.option("-s", "--samples", default=None, type=int,
              help="Sobol N (power of 2). Overrides calibration_conf value.")
@click.option("-c", "--cpus", default=-1, help="Cores (-1 for all).")
@click.pass_context
def run_sample(ctx, samples, cpus):
    """Generate Sobol sample and run simulations."""
    params_to_calibrate = calibration_conf.CALIBRATION_PARAMETERS
    settings            = calibration_conf.CALIBRATION_SETTINGS

    names = list(params_to_calibrate.keys())
    lbs   = [v[0] for v in params_to_calibrate.values()]
    ubs   = [v[1] for v in params_to_calibrate.values()]

    n_samples = samples if samples is not None else settings["samples"]
    n_runs    = settings["runs_per_sample"]

    if (n_samples & (n_samples - 1)) != 0:
        logger.warning(f"samples={n_samples} is not a power of 2.")

    problem = {
        "num_vars": len(names),
        "names":    names,
        "bounds":   list(zip(lbs, ubs))
    }
    scaled_samples = sobol_sampler.sample(
        problem,
        N=n_samples,
        seed=settings["sobol_seed"],
    )
    start_date = datetime.strptime(calibration_conf.CALIBRATION_SETTINGS['target_start_year'], '%Y-%m-%d').date()
    end_date = datetime.strptime(calibration_conf.CALIBRATION_SETTINGS['target_end_year'], '%Y-%m-%d').date()   
    processing_acp = {"PROCESSING_ACPS":[calibration_conf.CALIBRATION_SETTINGS['calibration_region']],
                      "STARTING_DAY":start_date,
                      "TOTAL_DAYS":(end_date-start_date).days}
    confs = [dict(zip(names, scaled_samples[i])) for i in range(len(scaled_samples))]
    for conf in confs: 
        conf.update(processing_acp)

    output_dir = gen_output_dir("calibration")

    _save_meta(output_dir, problem, n_samples, n_runs, settings)
    multiple_runs(confs, n_runs, cpus, output_dir)

    logger.info(f"Done. Results in: {output_dir}")


@main.command()
@click.argument("root_dir")
def score(root_dir):
    """Score an existing results folder."""
    score_calibration(root_dir)


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

@contextmanager
def _tqdm_joblib(tqdm_bar):
    """Patch joblib's batch callback so tqdm updates after each completed job."""
    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_bar.close()


def multiple_runs(overrides: list, runs: int, cpus: int, output_dir: str):
    """Dispatch all (parameter set × Monte Carlo run) jobs in parallel."""
    paths      = [os.path.join(output_dir, str(n)) for n in range(len(overrides))]
    param_list = []
    for o in overrides:
        p = copy.deepcopy(conf.PARAMS)
        p.update(o)
        param_list.append(p)

    job_specs = [
        {"path": os.path.join(path, str(i)), "params": p}
        for p, path in zip(param_list, paths)
        for i in range(runs)
    ]
    save_jobs(output_dir, job_specs, cpus)

    _dispatch(job_specs, cpus, desc="Sobol runs")
    logger.info("All runs completed.")


def single_run(params: dict, path: str):
    """Run one simulation and write outputs to path."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "conf.json"), "w") as f:
        json.dump({"PARAMS": params}, f, indent=4, default=str)
    
    sim = Simulation(params, path)
    sim.initialize()
    sim.run(log=False)
    open(os.path.join(path, "DONE"), "w").close()


# ── SCORING ───────────────────────────────────────────────────────────────────

def score_calibration(root_dir: str) -> pd.DataFrame:
    """
    Score each parameter set (mean across MC runs) and write
    calibration_scores.csv to root_dir.
    """
    tracked = list(calibration_conf.CALIBRATION_PARAMETERS.keys())
    results = []

    param_set_dirs = sorted([
        d for d in glob(os.path.join(root_dir, "*/"))
        if os.path.basename(os.path.normpath(d)).isdigit()
    ])

    for ps_dir in param_set_dirs:
        run_scores, param_snapshot = [], {}

        for rd in sorted(glob(os.path.join(ps_dir, "*/"))):
            if not os.path.basename(os.path.normpath(rd)).isdigit():
                continue

            conf_file = os.path.join(rd, "conf.json")
            if os.path.exists(conf_file):
                with open(conf_file) as f:
                    data = json.load(f)
                param_snapshot = {k: data["PARAMS"].get(k) for k in tracked}

            csv_path = os.path.join(rd, "stats.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(
                    csv_path,
                    names=OUTPUT_DATA_SPEC["stats"]["columns"],
                    sep=";",
                )
                run_scores.append(calculate_fitness(df))

        if run_scores:
            results.append({
                **param_snapshot,
                "score":     np.mean(run_scores),
                "score_std": np.std(run_scores),
                "n_runs":    len(run_scores),
                "path":      ps_dir,
            })

    score_df    = pd.DataFrame(results).sort_values("score").reset_index(drop=True)
    output_path = os.path.join(root_dir, "calibration_scores.csv")
    score_df.to_csv(output_path, index=False)

    logger.info(f"Scores saved to: {output_path}")
    print(score_df[tracked + ["score", "score_std"]].head(10).to_string(index=False))

    return score_df


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

    scores_path = os.path.join(root_dir, "calibration_scores.csv")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"calibration_scores.csv not found in {root_dir}. Run 'score' first.")

    problem   = meta["problem"]
    n_samples = meta["n_samples"]
    Y         = pd.read_csv(scores_path).sort_values("path")["score"].values

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

def _dispatch(job_specs: list, cpus: int, desc: str):
    """Run a list of {path, params} job specs in parallel with a progress bar."""
    jobs = [delayed(single_run)(job["params"], job["path"]) for job in job_specs]
    bar  = tqdm.tqdm(total=len(jobs), desc=desc, unit="sim", dynamic_ncols=True)
    with _tqdm_joblib(bar):
        Parallel(n_jobs=cpus, backend="multiprocessing")(jobs)


def _save_meta(output_dir: str, problem: dict, n_samples: int,
               n_runs: int, settings: dict):
    """Write meta.json required by compute_sensitivity."""
    meta = {
        "problem":   problem,
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