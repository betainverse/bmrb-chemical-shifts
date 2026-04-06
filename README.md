# BMRB Chemical Shifts

This repository pulls chemical-shift data from the BMRB website, organizes
that data into reusable local datasets, and generates several kinds of
graphical visualizations from those datasets.

The project is organized around three task areas:

1. `bmrb_data_cache/`
   BMRB data acquisition, parsing, caching, and local dataset building.
2. `methyl_heatmap_overlay/`
   Static 2D paired-shift heatmaps.
3. `carbon_proton_nitrogen_charts/`
   Cross-amino-acid charting for selected atom classes.

## BMRB Data Cache

Directory:
`bmrb_data_cache/`
(`chem_shifts/bmrb_data_cache`)

Use this area when you want to:

- fetch BMRB-derived source pages
- reorganize those pages into local cache files
- build reusable CSV datasets from BMRB-derived sources
- inspect or bulk-download BMRB histogram-derived data

### Standalone Task Scripts

`bmrb_shift_table.py`
(`chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/bmrb_shift_table.py`)

Builds the archive-wide atom-level BMRB chemical-shift CSV from BMRB entries.
This is the main script to run when you want a fresh master table of chemical
shifts without secondary-structure labels.

`fetch_all_filtered_histogram_pages.py`
(`chem_shifts/bmrb_data_cache/fetch_all_filtered_histogram_pages.py`)

Downloads all published filtered BMRB histogram pages into the local HTML
cache.

`fetch_all_ss_histogram_pages.py`
(`chem_shifts/bmrb_data_cache/fetch_all_ss_histogram_pages.py`)

Downloads all published BMRB secondary-structure histogram pages into the local
HTML cache.

`cached_html_to_csv_filtered.py`
(`chem_shifts/bmrb_data_cache/cached_html_to_csv_filtered.py`)

Converts cached filtered BMRB histogram HTML pages into the compact filtered
histogram CSV cache.

`cached_html_to_csv_ss.py`
(`chem_shifts/bmrb_data_cache/cached_html_to_csv_ss.py`)

Converts cached `_ss.html` BMRB histogram pages into the compact
secondary-structure histogram CSV cache.

`bmrb_ss_histogram_bulk_download.py`
(`chem_shifts/bmrb_data_cache/bmrb_ss_histogram_bulk_download.py`)

Bulk-downloads BMRB secondary-structure histogram data directly from published
`_ss.html` pages and writes combined CSV outputs.

`bmrb_ss_histogram_parser.py`
(`chem_shifts/bmrb_data_cache/bmrb_ss_histogram_parser.py`)

Extracts Plotly histogram data from a BMRB `_ss.html` page.

### Data Products In This Area

`filtered_histogram_cache_compact.csv`
(`chem_shifts/bmrb_data_cache/filtered_histogram_cache_compact.csv`)

Compact local cache derived from the published filtered BMRB histogram pages.
Each row stores one parsed histogram trace plus the serialized shift values for
that trace.
The raw filtered HTML cache used to build this file is kept locally but is not
versioned in git.

`ss_histogram_cache_compact.csv`
(`chem_shifts/bmrb_data_cache/ss_histogram_cache_compact.csv`)

Compact local cache derived from the published secondary-structure BMRB
histogram pages. Each row stores one parsed secondary-structure trace plus the
serialized shift values for that trace. The raw SS HTML cache used to build this
file is kept locally but is not versioned in git.

`bmrb_shift_table_full.csv`
(`chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/bmrb_shift_table_full.csv`)

Archive-wide atom-level chemical shift table built directly from BMRB entries.
Each row is one assigned chemical shift from the BMRB archive. This dataset
does not include secondary-structure labels, but it preserves assignment-level
identity information useful for pairing and correlation analyses.

Because this file is too large for standard GitHub version control, the clean
public repository is intended to keep the builder script while letting users
generate this CSV locally as needed.

### Important Data-Model Distinction

There are two different data models in active use:

- `bmrb_shift_table_full.csv` is an assignment-level table built directly from
  BMRB entry data. It preserves enough identity information to trace a chemical
  shift back to its BMRB entry, residue context, and atom assignment. This is
  the right dataset when a project needs to correlate one atom's shift with
  another atom from the same chemical entity.
- `filtered_histogram_cache_compact.csv` and
  `ss_histogram_cache_compact.csv` are compact histogram-cache tables built
  from the BMRB HTML histogram interface. They preserve histogram trace data
  and, in the SS case, BMRB-assigned secondary-structure-separated
  distributions. They do not preserve the per-assignment atom-to-atom linkage
  needed for direct within-entity correlation analyses.

## Static 2D Heatmaps For Paired Atom Chemical Shifts

Directory:
`methyl_heatmap_overlay/`
(`chem_shifts/methyl_heatmap_overlay`)

Use this area when you want static visualizations of paired atom chemical
shifts, especially proton/carbon pairs shown as 2D density or heatmap-style
plots.

`render_methyl_heatmaps_html.py`
(`chem_shifts/methyl_heatmap_overlay/render_methyl_heatmaps_html.py`)

Renders paired methyl proton/carbon heatmaps as standalone HTML.

`render_methyl_heatmaps_static.py`
(`chem_shifts/methyl_heatmap_overlay/render_methyl_heatmaps_static.py`)

Renders paired methyl proton/carbon heatmaps as static PDF and SVG outputs.

## Carbon, Proton, And Nitrogen Summary Charts

Directory:
`carbon_proton_nitrogen_charts/`
(`chem_shifts/carbon_proton_nitrogen_charts`)

Use this area when you want charts that compare selected atom classes across
all amino acids, especially for recognition or amino-acid identification from
chemical shifts.

These scripts read their compact input CSVs from:
`bmrb_data_cache/`
(`chem_shifts/bmrb_data_cache`)

They write generated outputs under:
`carbon_proton_nitrogen_charts/output/`
(`chem_shifts/carbon_proton_nitrogen_charts/output`)

`render_carbon_pages_from_cached_csv.py`
(`chem_shifts/carbon_proton_nitrogen_charts/render_carbon_pages_from_cached_csv.py`)

Builds carbon-focused multi-page summary charts from the compact filtered and
secondary-structure histogram caches.

`render_proton_page_from_cached_csv.py`
(`chem_shifts/carbon_proton_nitrogen_charts/render_proton_page_from_cached_csv.py`)

Builds proton-focused summary charts from the compact filtered and
secondary-structure histogram caches.

`render_nitrogen_page_from_cached_csv.py`
(`chem_shifts/carbon_proton_nitrogen_charts/render_nitrogen_page_from_cached_csv.py`)

Builds nitrogen-focused summary charts from the compact filtered and
secondary-structure histogram caches.

`render_combined_shift_pages_from_cached_csv.py`
(`chem_shifts/carbon_proton_nitrogen_charts/render_combined_shift_pages_from_cached_csv.py`)

Builds the combined multi-page carbon/proton/nitrogen chart document from the
compact caches. In this clean repository, the optional CYANA nomenclature image
is not included, so the special-cases page is still rendered but the lower
image panel remains blank unless you provide that image yourself.

`render_carbon_charts_interactive.py`
(`chem_shifts/carbon_proton_nitrogen_charts/render_carbon_charts_interactive.py`)

Builds an interactive HTML version of the carbon chart set from the compact
caches.
