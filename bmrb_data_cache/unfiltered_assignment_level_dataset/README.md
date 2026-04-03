# Unfiltered Assignment-Level Dataset

This directory holds the current assignment-level BMRB chemical-shift workflow:

- `bmrb_shift_table.py`
  (`chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/bmrb_shift_table.py`)
- `bmrb_shift_table_full.csv`
  (`chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/bmrb_shift_table_full.csv`)

What this workflow is for:

- building an assignment-level table directly from BMRB entry/API data
- preserving enough identity information to correlate one atom's chemical shift
  with another atom from the same chemical entity

Important limitations:

- the dataset is not filtered using the exclusion criteria described on the
  filtered BMRB histogram/statistics pages
- the dataset does not contain secondary-structure labels

Why this directory is separated:

- it keeps the current unfiltered assignment-level workflow grouped with its own
  generated CSV
- it makes clear that this workflow is distinct from the filtered histogram-page
  cache workflows elsewhere in `bmrb_data_cache/`

Replacement plan:

- the long-term goal is to replace this workflow with a filtered
  assignment-level dataset that preserves atom-to-atom correlation while also
  aligning more closely with the filtered BMRB histogram/statistics criteria
