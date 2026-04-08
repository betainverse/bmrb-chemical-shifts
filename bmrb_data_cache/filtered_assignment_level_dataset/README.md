# Filtered Assignment-Level Dataset

This directory holds the first phase of the filtered assignment-level workflow.

Current contents in this workflow:

- `build_filtered_entry_manifest.py`
  (`chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_entry_manifest.py`)
- `build_filtered_assignment_level_csv.py`
  (`chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_assignment_level_csv.py`)
- `bmrb_filtered_entry_manifest.csv`
  (`chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/bmrb_filtered_entry_manifest.csv`)
- `bmrb_shift_table_filtered.csv`
  (`chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/bmrb_shift_table_filtered.csv`)

What this workflow is for:

- start from the same BMRB API-driven entry universe used by the current
  unfiltered assignment-level builder
- build a complete entry-level inclusion/exclusion manifest for a filtered
  assignment-level dataset
- derive a filtered assignment-level CSV from the existing local unfiltered CSV
  and that manifest
- record best-fit local exclusion reasons that approximate the filtered BMRB
  histogram/statistics behavior described at
  `https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt`

What this workflow does not do:

- it does not rebuild the current unfiltered assignment-level CSV
- it does not attempt to handle non-numeric BMRB IDs such as `bmse000001`

Current pipeline:

1. use the same BMRB entry universe and cache conventions as:
   `unfiltered_assignment_level_dataset/bmrb_shift_table.py`
   (`chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/bmrb_shift_table.py`)
2. build the entry-level exclusion manifest in this directory
3. build the filtered assignment-level CSV from the existing unfiltered
   assignment-level CSV and the manifest rows marked `included`

Manifest notes:

- manifest rows are entry-level, not assignment-level
- both included and excluded BMRB entries are recorded
- exclusion reasons are best-fit local reasons, not a claim of exact BMRB
  internal filtering behavior

Filtered CSV build metadata:

- build date recorded from file timestamp: 2026-04-06 16:23:38
- row count: 10613382
