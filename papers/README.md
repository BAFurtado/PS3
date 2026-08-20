# Papers

Reproduction packages for papers built on PolicySpace3. Each subdirectory is a frozen
snapshot of a submitted or published version: manuscript source, the analysis-ready data
behind the results, the pipeline that produced them, and a link to the raw simulated
database where it is too large to commit.

| Package | Paper | Status |
|---|---|---|
| [`density_housing_inequality/`](density_housing_inequality/) | Housing policy and urban inequality across 27 Brazilian metropolitan areas | Submitted |

Each package has its own `README.md` with the download link for the simulated database and
step-by-step instructions. Run the pipeline scripts from the **project root**, not from
inside the package.

Snapshots are not edited in place. Working copies live in `text/`, and the analysis code
lives in `analysis/validation/`; each package carries a `sync_from_working.sh` that refreshes
it after a revision.
