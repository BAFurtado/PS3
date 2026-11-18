"""MCMV queue order test: Goiânia, d09 and d25, seeds 0-3 (SEED 1000-1003), 2010-2034, the runner_mcmv2026 settings
plus TOTAL_TARGETING_POLICY True (eligible households sorted by permanent income, poorest first). Compared with the
same runs of output/mcmv2026 (unsorted, queue in area-code order).
usage (repo root, env ps3_2): python analysis/mcmv2026/queue_test.py CPUS [FIRST LAST], seeds FIRST..LAST (0 3)
Output: output/queue_test/<scenario>/GOIANIA/<i>/; rerun the command to resume."""
import logging
import sys

sys.path.insert(0, '.')
import runner_mcmv2026 as M  # noqa: E402

R = M.R
ROOT = 'output/queue_test'

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    M.add_scenarios()
    first, last = (int(x) for x in sys.argv[2:4]) if len(sys.argv) > 3 else (0, 3)
    items = M.queue(['d09', 'd25'], ['GOIANIA'], list(range(first, last + 1)))
    R.run(ROOT, items, int(sys.argv[1]), ROOT, dict(M.COMMON, TOTAL_TARGETING_POLICY=True))
