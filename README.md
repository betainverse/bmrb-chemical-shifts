# BMRB Chemical Shifts

This repository pulls chemical-shift data from the BMRB website, organizes
that data into reusable local datasets, and generates several kinds of
graphical visualizations from those datasets.

The project is organized around four task areas:

1. `bmrb_data_cache/`
   BMRB data acquisition, parsing, caching, and local dataset building.
2. `methyl_heatmap_overlay/`
   Static 2D paired-shift heatmaps.
3. `methyl_heatmap_exploration/`
   Exploratory methyl heatmap parameter sweeps across multiple datasets and rendering settings.
4. `carbon_proton_nitrogen_charts/`
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

This area now contains two parallel assignment-level workflows:

- `unfiltered_assignment_level_dataset/`
  - the current assignment-level source workflow
- `filtered_assignment_level_dataset/`
  - the new filtered assignment-level workflow area
  - this workflow begins by building an entry-level exclusion manifest
  - it depends on the same BMRB API / entry universe used by the current
    unfiltered builder
  - it approximates the exclusion rules described by BMRB on the filtered
    statistics page: `https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt`
  - at present, this workflow only handles numeric BMRB entry IDs
  - non-numeric IDs returned by the BMRB entry listing, such as
    `bmse000001`, are skipped and are not represented in the manifest
  - the filtered assignment-level CSV is a later pipeline step, not part of
    this first phase

### Bootstrap Rebuild Sequence

From the repository root, the full local rebuild sequence for the HTML caches
and compact CSVs is:

```bash
python3 bmrb_data_cache/fetch_all_filtered_histogram_pages.py
python3 bmrb_data_cache/fetch_all_ss_histogram_pages.py
python3 bmrb_data_cache/cached_html_to_csv_filtered.py
python3 bmrb_data_cache/cached_html_to_csv_ss.py
```

This sequence:

- rebuilds the local filtered histogram HTML cache
- rebuilds the local SS histogram HTML cache
- rebuilds `filtered_histogram_cache_compact.csv`
- rebuilds `ss_histogram_cache_compact.csv`

The HTML caches are local-only and are not versioned in git.

The filtered assignment-level pipeline is:

1. obtain candidate BMRB entries using the same API-driven entry workflow as
   the unfiltered builder
2. build the filtered entry manifest
3. build the filtered assignment-level CSV by applying the manifest to the
   existing unfiltered assignment-level CSV

### Standalone Task Scripts

`bmrb_shift_table.py`
(`chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/bmrb_shift_table.py`)

Builds the archive-wide atom-level BMRB chemical-shift CSV from BMRB entries.
This is the main script to run when you want a fresh master table of chemical
shifts without secondary-structure labels.

`build_filtered_entry_manifest.py`
(`chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_entry_manifest.py`)

Builds the entry-level exclusion manifest for the filtered assignment-level
workflow. This is the first filtered artifact, and it records both included and
excluded BMRB entries using best-fit local exclusion rules that approximate the
criteria described by BMRB at
`https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt`. At present it only
handles numeric BMRB IDs, so non-numeric IDs such as `bmse000001` are skipped.

`build_filtered_assignment_level_csv.py`
(`chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_assignment_level_csv.py`)

Builds the filtered assignment-level CSV by keeping only rows from
`bmrb_shift_table_full.csv` whose `bmrb_id` is marked `included` in
`bmrb_filtered_entry_manifest.csv`. This script does not rebuild the unfiltered
CSV; it derives the filtered CSV directly from the existing local unfiltered
dataset.

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

Current committed snapshot metadata:

- build date recorded from file timestamp: 2026-03-31 15:03:14
- row count: 299

`ss_histogram_cache_compact.csv`
(`chem_shifts/bmrb_data_cache/ss_histogram_cache_compact.csv`)

Compact local cache derived from the published secondary-structure BMRB
histogram pages. Each row stores one parsed secondary-structure trace plus the
serialized shift values for that trace. The raw SS HTML cache used to build this
file is kept locally but is not versioned in git.

Current committed snapshot metadata:

- build date recorded from file timestamp: 2026-03-31 15:11:10
- row count: 1044

`bmrb_shift_table_full.csv`
(`chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/bmrb_shift_table_full.csv`)

Archive-wide atom-level chemical shift table built directly from BMRB entries.
Each row is one assigned chemical shift from the BMRB archive. This dataset
does not include secondary-structure labels, but it preserves assignment-level
identity information useful for pairing and correlation analyses.

Because this file is too large for standard GitHub version control, the clean
public repository is intended to keep the builder script while letting users
generate this CSV locally as needed.

`bmrb_filtered_entry_manifest.csv`
(`chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/bmrb_filtered_entry_manifest.csv`)

Entry-level inclusion/exclusion manifest for the filtered assignment-level
workflow. Each row records a BMRB entry decision, whether the
entry should be excluded from the filtered dataset, and the best-fit
local reason codes that triggered exclusion.

`bmrb_shift_table_filtered.csv`
(`chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/bmrb_shift_table_filtered.csv`)

Filtered assignment-level chemical shift table derived by applying
`bmrb_filtered_entry_manifest.csv` to the existing unfiltered
`bmrb_shift_table_full.csv`. The output keeps the same assignment-level row
structure as the unfiltered CSV, but only for entries marked `included` in the
manifest.

Current local build metadata:

- build date recorded from file timestamp: 2026-04-06 16:23:38
- row count: 10613382

Current comparison to the unfiltered assignment-level CSV:

- unfiltered size: 406.3 MB
- filtered size: 358.8 MB
- size reduction: 47.5 MB
- unfiltered assignment rows: 12021883
- filtered assignment rows: 10613382
- excluded assignment rows: 1408501
- included entries: 15996
- excluded entries: 1621

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
The production default uses a per-methyl-class relative-density threshold of
0.02. The script also keeps a clearly commented manual alternative in the code
for a `min-count = 5` render if that stronger cutoff is wanted later.

## Exploratory Methyl Heatmap Sweeps

Directory:
`methyl_heatmap_exploration/`
(`chem_shifts/methyl_heatmap_exploration`)

Use this area when you want to compare how methyl proton-carbon overlay
heatmaps look under different rendering settings.

`render_methyl_heatmap_parameter_sweep.py`
(`chem_shifts/methyl_heatmap_exploration/render_methyl_heatmap_parameter_sweep.py`)

Created 2026-04-06. Builds exploratory comparison sheets for both the
unfiltered and filtered assignment-level datasets by sweeping display-threshold
values at one or more heatmap bin sizes while keeping one fixed
`outlier_trim_percent` setting. Each panel is annotated with the dataset label,
bin size, `outlier_trim_percent`, threshold mode, and threshold value used so
the parameter choices can be compared visually. The code distinguishes three
display-threshold ideas: minimum count per bin, minimum density relative to the
maximum bin for each methyl class, and alpha cutoff. The current exploratory
default is relative-density = 0.02.

Output files are written to:
`methyl_heatmap_exploration/output/`
(`chem_shifts/methyl_heatmap_exploration/output`)

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
