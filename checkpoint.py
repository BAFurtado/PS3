"""
Checkpoint utilities for interruptible simulation runs.

Runners call save_jobs() before dispatching and write a DONE sentinel at the
end of each single_run(). On interruption, resume commands call pending_jobs()
to clean partial outputs and recover the remaining work.
"""
import json
import os
import shutil
from datetime import datetime


def save_jobs(output_dir: str, job_specs: list, cpus: int):
    """Write jobs.json to output_dir before dispatching."""
    manifest = {
        "created": datetime.now().isoformat(),
        "cpus":    cpus,
        "jobs":    job_specs,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "jobs.json"), "w") as f:
        json.dump(manifest, f, indent=4, default=str)


def pending_jobs(root_dir: str) -> tuple[list, int]:
    """
    Read jobs.json, remove directories of incomplete runs, and return
    (pending_job_specs, n_cleaned).
    """
    jobs_path = os.path.join(root_dir, "jobs.json")
    if not os.path.exists(jobs_path):
        raise FileNotFoundError(f"No jobs.json found in {root_dir}.")

    with open(jobs_path) as f:
        manifest = json.load(f)

    pending, cleaned = [], 0
    for job in manifest["jobs"]:
        path = job["path"]
        if os.path.exists(os.path.join(path, "DONE")):
            continue
        if os.path.exists(path):
            shutil.rmtree(path)
            cleaned += 1
        pending.append(job)

    return pending, cleaned
