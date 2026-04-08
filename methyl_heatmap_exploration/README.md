# Exploratory Methyl Heatmap Sweeps

This directory holds exploratory methyl overlay rendering work.

Current contents:

- `render_methyl_heatmap_parameter_sweep.py`
  (`chem_shifts/methyl_heatmap_exploration/render_methyl_heatmap_parameter_sweep.py`)
- `output/`
  (`chem_shifts/methyl_heatmap_exploration/output`)

What this workflow is for:

- compare filtered and unfiltered methyl overlay heatmaps side by side
- sweep across display-threshold values at one or more heatmap bin sizes while keeping one fixed outlier-trim setting
- annotate each panel with the exact rendering settings used, including threshold mode and threshold value
- keep `outlier_trim_percent` separate from future display-threshold concepts like minimum count, minimum density, and alpha cutoff
- help decide which parameter combinations look best before updating the main
  methyl heatmap workflow

Current default sweep:

- datasets: unfiltered and filtered assignment-level CSVs
- bin sizes: 120 by default
- outlier_trim_percent: 0.0 by default
- threshold mode: relative-density by default
- threshold values: 0.02 by default

Outputs are written into `output/` as comparison-sheet PDFs and optional PNGs.
