"""
Checkpoint utilities for interruptible simulation runs.

Runners call save_jobs() before dispatching and write a DONE sentinel at the
end of each single_run(). On interruption, resume commands call pending_jobs()
to clean partial outputs and recover the remaining work.
"""
import json
import os
import re
import shutil
import datetime

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _restore_params(params: dict) -> dict:
    """Convert ISO date strings back to datetime.date after JSON round-trip."""
    out = {}
    for k, v in params.items():
        if isinstance(v, str) and _DATE_RE.match(v):
            try:
                out[k] = datetime.date.fromisoformat(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


def save_jobs(output_dir: str, job_specs: list, cpus: int):
    """Write jobs.json to output_dir before dispatching."""
    manifest = {
        "created": datetime.datetime.now().isoformat(),
        "cpus":    cpus,
        "jobs":    job_specs,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "jobs.json"), "w") as f:
        json.dump(manifest, f, indent=4, default=str)


def _run_index(path: str):
    """Replication index of a run, from its folder name, or None if not numeric."""
    name = os.path.basename(os.path.normpath(path))
    return int(name) if name.isdigit() else None


def _seed_base(jobs: list):
    """
    Common base of finished runs whose seed is base + replication index, or None.

    Finished runs must span at least two replication indices and all agree on one base.
    """
    bases, indices = set(), set()
    for job in jobs:
        conf_path = os.path.join(job["path"], "conf.json")
        index = _run_index(job["path"])
        if index is None or not os.path.exists(os.path.join(job["path"], "DONE")) \
                or not os.path.exists(conf_path):
            continue
        with open(conf_path) as f:
            seed = json.load(f)["PARAMS"].get("SEED")
        if seed is None:
            return None
        bases.add(int(seed) - index)
        indices.add(index)
    return bases.pop() if len(bases) == 1 and len(indices) > 1 else None


def pending_jobs(root_dir: str) -> tuple[list, int]:
    """
    Read jobs.json, remove directories of incomplete runs, and return
    (pending_job_specs, n_cleaned).

    A pending job without a SEED, from a manifest written before seeds were stored,
    gets base + replication index when the finished runs follow that rule; otherwise
    its seed is resolved at run time as before.
    """
    jobs_path = os.path.join(root_dir, "jobs.json")
    if not os.path.exists(jobs_path):
        raise FileNotFoundError(f"No jobs.json found in {root_dir}.")

    with open(jobs_path) as f:
        manifest = json.load(f)

    base = None
    if any("SEED" not in job["params"] for job in manifest["jobs"]):
        base = _seed_base(manifest["jobs"])

    pending, cleaned = [], 0
    for job in manifest["jobs"]:
        if base is not None and "SEED" not in job["params"] and _run_index(job["path"]) is not None:
            job["params"]["SEED"] = base + _run_index(job["path"])
        path = job["path"]
        if os.path.exists(os.path.join(path, "DONE")):
            continue
        if os.path.exists(path):
            shutil.rmtree(path)
            cleaned += 1
        job["params"] = _restore_params(job["params"])
        pending.append(job)

    return pending, cleaned
