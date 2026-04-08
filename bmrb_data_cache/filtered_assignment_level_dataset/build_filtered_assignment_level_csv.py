#!/usr/bin/env python3
"""
Created: 2026-04-06

Purpose:
- build a filtered assignment-level chemical-shift CSV by applying the
  entry-level exclusion manifest to the existing unfiltered assignment-level CSV
- preserve the assignment-level rows for entries marked `included` in the
  manifest
- provide the next pipeline step after `build_filtered_entry_manifest.py`

Expected input data:
- `bmrb_shift_table_full.csv` in
  `chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/`
- `bmrb_filtered_entry_manifest.csv` in
  `chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/`
- the manifest is expected to contain one row per numeric BMRB entry ID
- non-numeric BMRB IDs such as `bmse000001` are not handled by this workflow,
  because they are intentionally skipped when the manifest is built

Expected output data:
- `bmrb_shift_table_filtered.csv` in
  `chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/`
- the output CSV keeps the same column structure as the unfiltered
  assignment-level CSV, but only includes rows whose `bmrb_id` is marked
  `included` in the manifest

Relationship to other workflows:
- this script does not rebuild the unfiltered assignment-level CSV
- this script depends on the filtered manifest built by
  `build_filtered_entry_manifest.py`
- together, the unfiltered builder, manifest builder, and this script define the
  current filtered assignment-level pipeline
- the filtering logic is intended to approximate the BMRB filtered statistics
  behavior described at
  `https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt`

Command-line use:
- default invocation:
  `python3 bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_assignment_level_csv.py`
- common options:
  `--input-csv` to point at a different unfiltered assignment-level CSV
  `--manifest-csv` to use a different manifest file
  `--output-csv` to write the filtered CSV to a different location
- example using the default local pipeline files:
  `python3 bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_assignment_level_csv.py`
- example writing to a temporary output file:
  `python3 bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_assignment_level_csv.py --output-csv /tmp/bmrb_shift_table_filtered.csv`
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / 'bmrb_data_cache'
    / 'unfiltered_assignment_level_dataset'
    / 'bmrb_shift_table_full.csv'
)
DEFAULT_MANIFEST_CSV = Path(__file__).resolve().with_name('bmrb_filtered_entry_manifest.csv')
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().with_name('bmrb_shift_table_filtered.csv')
EXPECTED_ASSIGNMENT_COLUMNS = [
    'bmrb_id',
    'entity_id',
    'entity_assembly_id',
    'comp_index_id',
    'residue_3',
    'residue_1',
    'atom',
    'element',
    'shift_ppm',
    'ambiguity_code',
    'assigned_chem_shift_list_id',
]
EXPECTED_MANIFEST_COLUMNS = [
    'bmrb_id',
    'manifest_status',
    'exclude_from_filtered_dataset',
    'reason_codes',
    'reason_summary',
    'ligand_flag',
    'paramagnetic_or_aromatic_ligand_flag',
    'eight_sigma_outlier_flag',
    'carbon_bound_proton_extreme_flag',
    'withdrawn_entry_flag',
    'rule_version',
    'built_at',
]


@dataclass
class FilterRunSummary:
    input_row_count: int
    written_row_count: int
    included_entry_count: int
    excluded_entry_count: int
    matched_included_entry_count: int
    missing_included_entry_count: int


# Parse command-line arguments for the standalone CSV-filtering step.
#
# Input data shape:
# - raw CLI arguments from the shell
# - optional paths to the unfiltered assignment-level CSV, the manifest CSV,
#   and the filtered output CSV
#
# Output data shape:
# - an argparse.Namespace with `input_csv`, `manifest_csv`, and `output_csv`
#   Path objects

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a filtered assignment-level CSV from the unfiltered CSV and entry manifest.'
    )
    parser.add_argument(
        '--input-csv',
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help='Path to the unfiltered assignment-level CSV. '
        f'Default: {DEFAULT_INPUT_CSV}',
    )
    parser.add_argument(
        '--manifest-csv',
        type=Path,
        default=DEFAULT_MANIFEST_CSV,
        help='Path to the filtered entry manifest CSV. '
        f'Default: {DEFAULT_MANIFEST_CSV}',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help='Path for the filtered assignment-level CSV. '
        f'Default: {DEFAULT_OUTPUT_CSV}',
    )
    return parser.parse_args()


# Ensure that an expected input file exists before the pipeline starts.
#
# Input data shape:
# - `path`: Path to a required file
# - `label`: short human-readable name used in the error message
#
# Output data shape:
# - no return value
# - raises FileNotFoundError if the file is missing

def ensure_file_exists(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')


# Check that a CSV file has exactly the columns the filtering pipeline expects.
#
# Input data shape:
# - `fieldnames`: sequence of column names read from csv.DictReader
# - `expected_columns`: ordered list of required column names
# - `label`: short human-readable name used in the error message
#
# Output data shape:
# - no return value
# - raises ValueError if the columns do not match exactly

def validate_columns(fieldnames: list[str] | None, expected_columns: list[str], label: str) -> None:
    actual_columns = list(fieldnames or [])
    if actual_columns != expected_columns:
        raise ValueError(
            f'{label} columns do not match expected schema. '            f'Expected {expected_columns}, found {actual_columns}'
        )


# Load the manifest and collect the BMRB entry IDs that should remain in the
# filtered assignment-level CSV.
#
# Input data shape:
# - `manifest_csv`: CSV with one row per BMRB entry and the manifest columns
#   described in EXPECTED_MANIFEST_COLUMNS
#
# Output data shape:
# - a tuple containing:
#   - a set of included numeric `bmrb_id` strings
#   - the total count of included entries in the manifest
#   - the total count of excluded entries in the manifest

def load_included_entry_ids(manifest_csv: Path) -> tuple[set[str], int, int]:
    included_entry_ids: set[str] = set()
    included_count = 0
    excluded_count = 0

    with manifest_csv.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames, EXPECTED_MANIFEST_COLUMNS, 'Manifest CSV')
        for row in reader:
            bmrb_id = (row['bmrb_id'] or '').strip()
            status = (row['manifest_status'] or '').strip()
            exclude_flag = (row['exclude_from_filtered_dataset'] or '').strip().lower()
            if not bmrb_id:
                raise ValueError('Manifest CSV contains a row with an empty bmrb_id')
            if status == 'included' and exclude_flag == 'no':
                included_entry_ids.add(bmrb_id)
                included_count += 1
            elif status == 'excluded' and exclude_flag == 'yes':
                excluded_count += 1
            else:
                raise ValueError(
                    'Manifest CSV contains a row with inconsistent inclusion fields: '
                    f'bmrb_id={bmrb_id!r}, manifest_status={status!r}, '
                    f'exclude_from_filtered_dataset={exclude_flag!r}'
                )

    return included_entry_ids, included_count, excluded_count


# Stream the unfiltered assignment-level CSV and write only rows whose BMRB
# entry IDs are marked included in the manifest.
#
# Input data shape:
# - `input_csv`: assignment-level CSV with the columns in
#   EXPECTED_ASSIGNMENT_COLUMNS
# - `output_csv`: destination path for the filtered assignment-level CSV
# - `included_entry_ids`: set of manifest-approved `bmrb_id` strings
# - `included_entry_count`: total number of included entries in the manifest
# - `excluded_entry_count`: total number of excluded entries in the manifest
#
# Output data shape:
# - a FilterRunSummary describing how many rows were read and written, and how
#   many included entries were actually represented in the unfiltered CSV

def filter_assignment_rows(
    input_csv: Path,
    output_csv: Path,
    included_entry_ids: set[str],
    included_entry_count: int,
    excluded_entry_count: int,
) -> FilterRunSummary:
    input_row_count = 0
    written_row_count = 0
    matched_included_entries: set[str] = set()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open(newline='', encoding='utf-8') as in_handle, output_csv.open(
        'w', newline='', encoding='utf-8'
    ) as out_handle:
        reader = csv.DictReader(in_handle)
        validate_columns(reader.fieldnames, EXPECTED_ASSIGNMENT_COLUMNS, 'Assignment CSV')
        writer = csv.DictWriter(out_handle, fieldnames=EXPECTED_ASSIGNMENT_COLUMNS)
        writer.writeheader()

        for row in reader:
            input_row_count += 1
            bmrb_id = (row['bmrb_id'] or '').strip()
            if bmrb_id in included_entry_ids:
                writer.writerow({column: row[column] for column in EXPECTED_ASSIGNMENT_COLUMNS})
                written_row_count += 1
                matched_included_entries.add(bmrb_id)

    missing_included_entry_count = included_entry_count - len(matched_included_entries)
    return FilterRunSummary(
        input_row_count=input_row_count,
        written_row_count=written_row_count,
        included_entry_count=included_entry_count,
        excluded_entry_count=excluded_entry_count,
        matched_included_entry_count=len(matched_included_entries),
        missing_included_entry_count=missing_included_entry_count,
    )


# Print a compact human-readable summary after the filtered CSV is written.
#
# Input data shape:
# - `summary`: FilterRunSummary with row counts and entry counts from the run
# - `output_csv`: Path to the filtered assignment-level CSV that was written
#
# Output data shape:
# - no return value
# - writes a few summary lines to stdout for the person running the script

def print_summary(summary: FilterRunSummary, output_csv: Path) -> None:
    print(f'Wrote filtered assignment-level CSV to {output_csv}')
    print(f'Input assignment rows read: {summary.input_row_count}')
    print(f'Filtered assignment rows written: {summary.written_row_count}')
    print(f'Manifest included entries: {summary.included_entry_count}')
    print(f'Manifest excluded entries: {summary.excluded_entry_count}')
    print(f'Included entries matched in assignment CSV: {summary.matched_included_entry_count}')
    print(f'Included entries missing from assignment CSV: {summary.missing_included_entry_count}')


# Run the filtered assignment-level CSV pipeline end to end.
#
# Input data shape:
# - no direct input arguments; reads CLI arguments and the two CSV inputs
#
# Output data shape:
# - writes the filtered assignment-level CSV to disk
# - prints a short run summary to stdout
# - exits by raising exceptions if required files or schemas are invalid

def main() -> None:
    args = parse_args()
    ensure_file_exists(args.input_csv, 'Unfiltered assignment-level CSV')
    ensure_file_exists(args.manifest_csv, 'Filtered entry manifest CSV')
    included_entry_ids, included_count, excluded_count = load_included_entry_ids(args.manifest_csv)
    summary = filter_assignment_rows(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        included_entry_ids=included_entry_ids,
        included_entry_count=included_count,
        excluded_entry_count=excluded_count,
    )
    print_summary(summary, args.output_csv)


if __name__ == '__main__':
    main()
