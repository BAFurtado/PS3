"""
Figures for the emissions paper (runner_emissions.py batches).

Modules:
    data           reads a batch (stats.csv, firms.parquet, agents.parquet) into processed tables, cached in
                   <root>/_figcache/; matched comparisons against NO_POLICY
    decomposition  LMDI split of an emissions change into intensity (innovation) and scale (contraction)
    style          colours, fonts and panel helpers shared by every figure
    figures        every figure of the paper, one function each, in the order of the results section
    figures.ipynb  preview: loads a batch and draws each figure in its own cell

Run from the repository root:
    python -m analysis.emissions_paper.figures [ROOT] [--out DIR] [--outline SHP]
"""
