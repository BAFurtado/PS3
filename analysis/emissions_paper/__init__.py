"""
Figures for the emissions paper (runner_emissions.py batches).

Modules:
    style          colours, fonts and panel helpers shared by every figure
    data           reads a batch (stats.csv, firms.parquet) into tidy tables, cached in <root>/_figcache/
    decomposition  LMDI split of an emissions change into intensity (innovation) and scale (contraction)
    fig_main       the main figure: paths, map, sector mix, instruments

Run from the repository root:
    python -m analysis.emissions_paper.fig_main [ROOT] [--out DIR] [--window 2025 2029]
"""
