#!/usr/bin/env python3
"""
Created: 2026-04-06

Purpose:
- generate exploratory batches of methyl proton-carbon overlay heatmaps from
  both the filtered and unfiltered assignment-level CSV datasets
- compare display-threshold behavior at one or more heatmap bin sizes while
  keeping one explicit outlier-trimming setting fixed across the batch
- annotate every rendered heatmap with the dataset label, bin size,
  outlier-trim setting, threshold mode, and threshold value used for that panel

Expected input data:
- `bmrb_shift_table_full.csv` in
  `chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/`
- `bmrb_shift_table_filtered.csv` in
  `chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/`
- both CSVs are expected to use the assignment-level columns:
  `bmrb_id, entity_id, entity_assembly_id, comp_index_id, residue_3, residue_1, atom, element, shift_ppm, ambiguity_code, assigned_chem_shift_list_id`
- the filtered CSV only covers numeric BMRB IDs included by the manifest-based
  filtered workflow; non-numeric IDs such as `bmse000001` are outside the scope
  of this exploratory script

Expected output data:
- comparison-sheet PDFs in `chem_shifts/methyl_heatmap_exploration/output/`
- optional PNG companions in the same output directory
- each output page contains a grid of heatmaps covering one dataset across the
  requested threshold values and heatmap bin sizes

Relationship to other workflows:
- this script is an exploratory companion to the production scripts in
  `methyl_heatmap_overlay/`
- it does not rebuild the filtered or unfiltered assignment-level CSVs
- it reuses the same residue-pairing logic as the methyl overlay workflow, but
  renders many parameter combinations in one batch for visual comparison
- the filtered dataset used here ultimately depends on the BMRB filtering logic
  approximated from `https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt`

Command-line use:
- default invocation:
  `python3 methyl_heatmap_exploration/render_methyl_heatmap_parameter_sweep.py`
- common options:
  `--unfiltered-csv` to point at a different unfiltered assignment-level CSV
  `--filtered-csv` to point at a different filtered assignment-level CSV
  `--bins` to choose one or more heatmap bin counts
  `--outlier-trim-percent` to remove extreme proton/carbon values before binning
  `--threshold-mode` to choose the display-threshold rule
  `--threshold-values` to choose one or more threshold values for that mode
  `--output-dir` to write the comparison sheets somewhere else
  `--write-png` to also export PNG copies alongside the PDFs
- example sweeping relative-density thresholds at 120 bins:
  `python3 methyl_heatmap_exploration/render_methyl_heatmap_parameter_sweep.py --bins 120 --threshold-mode relative-density --threshold-values 0.00 0.02 0.05 0.10 --write-png`
- example sweeping minimum-count thresholds:
  `python3 methyl_heatmap_exploration/render_methyl_heatmap_parameter_sweep.py --bins 120 --threshold-mode min-count --threshold-values 0 1 2 3`
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLORATION_DIR = PROJECT_ROOT / 'methyl_heatmap_exploration'
DEFAULT_OUTPUT_DIR = EXPLORATION_DIR / 'output'
DEFAULT_UNFILTERED_CSV = (
    PROJECT_ROOT / 'bmrb_data_cache' / 'unfiltered_assignment_level_dataset' / 'bmrb_shift_table_full.csv'
)
DEFAULT_FILTERED_CSV = (
    PROJECT_ROOT / 'bmrb_data_cache' / 'filtered_assignment_level_dataset' / 'bmrb_shift_table_filtered.csv'
)
DEFAULT_BINS = [120]
DEFAULT_OUTLIER_TRIM_PERCENT = 0.0
DEFAULT_THRESHOLD_MODE = 'relative-density'
DEFAULT_THRESHOLD_VALUES = [0.02]
GLOBAL_X_RANGE = [-0.5, 3.0]
GLOBAL_Y_RANGE = [8.0, 30.0]
RESIDUE_KEY_FIELDS = (
    'bmrb_id',
    'entity_id',
    'entity_assembly_id',
    'comp_index_id',
    'assigned_chem_shift_list_id',
    'residue_3',
)
TYPE_COLORS = {
    'ala-hb-cb': '#e11d48',
    'ile-hd1-cd1': '#0f766e',
    'ile-hg2-cg2': '#65a30d',
    'leu-hd-cd': '#8b5cf6',
    'val-hg-cg': '#2563eb',
    'met-he-ce': '#c026d3',
    'thr-hg2-cg2': '#c2410c',
}
THRESHOLD_MODE_LABELS = {
    'relative-density': 'relative_density',
    'min-count': 'min_count',
    'alpha-cutoff': 'alpha_cutoff',
}
DEFAULT_CONTOUR_LEVELS = [0.15, 0.30, 0.50, 0.70]


@dataclass(frozen=True)
class PairVariant:
    label: str
    proton_atoms: tuple[str, ...]
    carbon_atoms: tuple[str, ...]


@dataclass(frozen=True)
class HeatmapSpec:
    slug: str
    title: str
    residue: str
    variants: tuple[PairVariant, ...]


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    label: str
    csv_path: Path


SPECS = (
    HeatmapSpec('ala-hb-cb', 'Alanine HB/CB', 'ALA', (PairVariant('ALA HB-CB', ('HB1', 'HB2', 'HB3'), ('CB',)),)),
    HeatmapSpec('ile-hd1-cd1', 'Isoleucine HD1/CD1', 'ILE', (PairVariant('ILE HD1-CD1', ('HD11', 'HD12', 'HD13'), ('CD1',)),)),
    HeatmapSpec('ile-hg2-cg2', 'Isoleucine HG2/CG2', 'ILE', (PairVariant('ILE HG2-CG2', ('HG21', 'HG22', 'HG23'), ('CG2',)),)),
    HeatmapSpec('val-hg-cg', 'Valine HG/CG', 'VAL', (
        PairVariant('VAL HG1-CG1', ('HG11', 'HG12', 'HG13'), ('CG1',)),
        PairVariant('VAL HG2-CG2', ('HG21', 'HG22', 'HG23'), ('CG2',)),
    )),
    HeatmapSpec('leu-hd-cd', 'Leucine HD/CD', 'LEU', (
        PairVariant('LEU HD1-CD1', ('HD11', 'HD12', 'HD13'), ('CD1',)),
        PairVariant('LEU HD2-CD2', ('HD21', 'HD22', 'HD23'), ('CD2',)),
    )),
    HeatmapSpec('met-he-ce', 'Methionine HE/CE', 'MET', (PairVariant('MET HE-CE', ('HE1', 'HE2', 'HE3'), ('CE',)),)),
    HeatmapSpec('thr-hg2-cg2', 'Threonine HG2/CG2', 'THR', (PairVariant('THR HG2-CG2', ('HG21', 'HG22', 'HG23'), ('CG2',)),)),
)

plt.rcParams.update(
    {
        'font.family': 'DejaVu Sans',
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.edgecolor': '#cbd5e1',
        'axes.labelcolor': '#0f172a',
        'xtick.color': '#334155',
        'ytick.color': '#334155',
    }
)


# Parse the script CLI so this exploratory sweep can be run directly.
#
# Input data shape:
# - shell arguments describing the two source CSV files, the bin sizes,
#   one fixed outlier-trimming percentage, one display-threshold mode, one or
#   more threshold values for that mode, and output-writing options
#
# Output data shape:
# - argparse.Namespace containing Path objects for the source/output locations,
#   Python lists for `bins` and `threshold_values`, one float for
#   `outlier_trim_percent`, and a mode string for `threshold_mode`
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render exploratory methyl heatmap parameter sweeps for filtered and unfiltered datasets.'
    )
    parser.add_argument('--unfiltered-csv', type=Path, default=DEFAULT_UNFILTERED_CSV)
    parser.add_argument('--filtered-csv', type=Path, default=DEFAULT_FILTERED_CSV)
    parser.add_argument('--bins', type=int, nargs='+', default=DEFAULT_BINS)
    parser.add_argument('--outlier-trim-percent', type=float, default=DEFAULT_OUTLIER_TRIM_PERCENT)
    parser.add_argument(
        '--threshold-mode',
        choices=['relative-density', 'min-count', 'alpha-cutoff'],
        default=DEFAULT_THRESHOLD_MODE,
    )
    parser.add_argument('--threshold-values', type=float, nargs='+', default=DEFAULT_THRESHOLD_VALUES)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--write-png', action='store_true')
    return parser.parse_args()


# Confirm that a required input file exists before attempting a sweep build.
#
# Input data shape:
# - `path`: Path to a CSV file that must exist
# - `label`: short human-readable label for error messages
#
# Output data shape:
# - no return value
# - raises FileNotFoundError if the expected file is missing
def ensure_file_exists(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')


# Read an assignment-level shift CSV into a list of row dictionaries.
#
# Input data shape:
# - `path`: CSV with assignment-level columns including residue identity,
#   atom name, and numeric shift value stored as text
#
# Output data shape:
# - list of dict[str, str], one dictionary per assignment-level CSV row
def read_shift_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


# Group assignment-level rows by residue instance and atom name so proton/carbon
# pairings can be reconstructed for methyl heatmaps.
#
# Input data shape:
# - `rows`: assignment-level row dictionaries read from the filtered or
#   unfiltered CSV
#
# Output data shape:
# - dictionary keyed by residue identity tuple, where each value is another
#   dictionary mapping atom names to lists of float shift values
def build_residue_atom_values(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, list[float]]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        atom = row['atom'].upper()
        residue = row['residue_3'].upper()
        if residue not in {'ALA', 'ILE', 'LEU', 'VAL', 'MET', 'THR'}:
            continue
        if not atom.startswith(('H', 'C')):
            continue
        try:
            shift = float(row['shift_ppm'])
        except ValueError:
            continue
        key = tuple(row[field] for field in RESIDUE_KEY_FIELDS)
        grouped[key][atom].append(shift)
    return grouped


# Convert grouped residue-level atom shifts into paired methyl proton/carbon
# points for one heatmap specification.
#
# Input data shape:
# - `grouped`: residue-level atom map from build_residue_atom_values
# - `spec`: HeatmapSpec describing which residue and atom variants to pair
#
# Output data shape:
# - list of `(x_ppm, y_ppm, label)` tuples, one point per residue instance that
#   provides both the requested proton and carbon shifts
def extract_points(
    grouped: dict[tuple[str, ...], dict[str, list[float]]], spec: HeatmapSpec
) -> list[tuple[float, float, str]]:
    points: list[tuple[float, float, str]] = []
    for residue_key, atom_map in grouped.items():
        if residue_key[-1] != spec.residue:
            continue
        for variant in spec.variants:
            proton_values = [value for atom in variant.proton_atoms for value in atom_map.get(atom, [])]
            carbon_values = [value for atom in variant.carbon_atoms for value in atom_map.get(atom, [])]
            if proton_values and carbon_values:
                points.append((mean(proton_values), mean(carbon_values), variant.label))
    return points


# Trim extreme points symmetrically from both axes using a percentile threshold.
#
# This is an input-data filter, not a heatmap-display threshold. It removes
# unusual paired points before any histogram bins are computed.
#
# Input data shape:
# - `points`: list of `(x_ppm, y_ppm, label)` methyl-pair tuples
# - `percentile`: percentage to trim from each tail on both x and y axes
#
# Output data shape:
# - filtered list of points that fall within the percentile bounds
# - if `percentile` is 0 or the input is empty, the original point list is
#   returned unchanged
def trim_outliers(points: list[tuple[float, float, str]], percentile: float) -> list[tuple[float, float, str]]:
    if not points or percentile <= 0.0:
        return list(points)
    xs = np.array([x for x, _, _ in points], dtype=float)
    ys = np.array([y for _, y, _ in points], dtype=float)
    x_low, x_high = np.percentile(xs, [percentile, 100.0 - percentile])
    y_low, y_high = np.percentile(ys, [percentile, 100.0 - percentile])
    return [(x, y, label) for x, y, label in points if x_low <= x <= x_high and y_low <= y <= y_high]


# Build a 2D histogram grid for one methyl overlay dataset.
#
# This is where a future minimum-count-per-bin threshold naturally starts,
# because raw histogram counts exist at this stage.
#
# Input data shape:
# - `points`: list of `(x_ppm, y_ppm, label)` tuples
# - `bins`: integer number of bins to use on each axis
#
# Output data shape:
# - tuple of `(x_centers, y_centers, z_grid)` numpy arrays suitable for imshow
#   where `z_grid` contains the transposed 2D histogram counts
def build_density_grid(points: list[tuple[float, float, str]], bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array([x for x, _, _ in points], dtype=float)
    ys = np.array([y for _, y, _ in points], dtype=float)
    x_edges = np.linspace(GLOBAL_X_RANGE[0], GLOBAL_X_RANGE[1], bins + 1)
    y_edges = np.linspace(GLOBAL_Y_RANGE[0], GLOBAL_Y_RANGE[1], bins + 1)
    hist, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    return x_centers, y_centers, hist.T


# Apply an absolute or relative threshold directly to one methyl-class density
# grid before it is converted to color and alpha.
#
# Important distinction:
# - `min-count` works on absolute raw counts
# - `relative-density` works on a fraction of this class's own maximum bin,
#   not on the global maximum across the whole overlaid panel
# - `alpha-cutoff` is not handled here because it acts later, after the density
#   grid has already been converted into display alpha
#
# Input data shape:
# - `z_grid`: 2D numpy array of histogram counts for one methyl class
# - `threshold_mode`: one of `relative-density`, `min-count`, or `alpha-cutoff`
# - `threshold_value`: float threshold value interpreted according to the mode
#
# Output data shape:
# - 2D numpy array of histogram counts with bins zeroed out when they fall
#   below the selected count-based or relative-density threshold
def apply_density_threshold(z_grid: np.ndarray, threshold_mode: str, threshold_value: float) -> np.ndarray:
    filtered = np.array(z_grid, copy=True)
    if threshold_mode == 'min-count':
        filtered[filtered < threshold_value] = 0.0
        return filtered
    if threshold_mode == 'relative-density':
        class_max = float(filtered.max())
        if class_max <= 0.0:
            return filtered
        filtered[filtered < (class_max * threshold_value)] = 0.0
        return filtered
    return filtered


# Convert a density grid into an RGBA image tinted to a methyl-class color.
#
# Three different threshold ideas matter here and should not be conflated:
# - minimum count per bin: absolute cutoff on raw histogram counts
# - minimum density relative to the max: cutoff based on a fraction of the
#   highest-density bin for this methyl class
# - alpha cutoff: cutoff on the final display alpha after the density-to-alpha
#   transform has already been applied
#
# This function handles the alpha-cutoff case. The other two threshold types are
# expected to be applied to the raw density grid before this function is called.
#
# Input data shape:
# - `hex_color`: color string such as `#2563eb`
# - `z_grid`: 2D numpy array of histogram counts, already thresholded if a
#   count-based or relative-density threshold is active
# - `threshold_mode`: one of `relative-density`, `min-count`, or `alpha-cutoff`
# - `threshold_value`: float threshold value interpreted according to the mode
#
# Output data shape:
# - 3D numpy array with shape `(height, width, 4)` for matplotlib imshow,
#   where the alpha channel scales with local density and may be clipped by an
#   alpha cutoff when that mode is selected
def density_rgba(
    hex_color: str,
    z_grid: np.ndarray,
    threshold_mode: str,
    threshold_value: float,
) -> np.ndarray:
    rgb = np.array(to_rgb(hex_color), dtype=float)
    if float(z_grid.max()) <= 0:
        norm = np.zeros_like(z_grid, dtype=float)
    else:
        norm = z_grid / float(z_grid.max())
    alpha = np.clip(norm ** 0.55, 0.0, 1.0) * 0.90
    if threshold_mode == 'alpha-cutoff':
        alpha[alpha < threshold_value] = 0.0
    rgba = np.zeros(z_grid.shape + (4,), dtype=float)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return rgba


# Build trimmed methyl overlay points for every heatmap spec from one already
# grouped dataset.
#
# Input data shape:
# - `grouped`: residue-level atom map from build_residue_atom_values
# - `outlier_trim_percent`: percentile trim applied to paired points before
#   binning begins
#
# Output data shape:
# - list of `(HeatmapSpec, points)` tuples where points are trimmed methyl pair
#   coordinates for that spec
# - raises RuntimeError if any expected methyl class has no points
def build_spec_points_from_grouped(
    grouped: dict[tuple[str, ...], dict[str, list[float]]],
    outlier_trim_percent: float,
) -> list[tuple[HeatmapSpec, list[tuple[float, float, str]]]]:
    spec_points = [
        (spec, trim_outliers(extract_points(grouped, spec), outlier_trim_percent))
        for spec in SPECS
    ]
    missing = [spec.title for spec, points in spec_points if not points]
    if missing:
        raise RuntimeError(f'No paired points found for: {", ".join(missing)}')
    return spec_points


# Build normalized contour levels for one methyl-class density grid.
#
# Input data shape:
# - `threshold_mode`: active display-threshold mode
# - `threshold_value`: active threshold value for that mode
#
# Output data shape:
# - list of normalized contour levels between 0 and 1 used to outline stronger
#   regions of the density map
# - when relative-density thresholding is active, contour levels at or below the
#   active threshold are skipped because those bins were already suppressed

def build_contour_levels(threshold_mode: str, threshold_value: float) -> list[float]:
    if threshold_mode == 'relative-density':
        return [level for level in DEFAULT_CONTOUR_LEVELS if level > threshold_value]
    return list(DEFAULT_CONTOUR_LEVELS)


# Smooth a normalized density grid before contour extraction so contour
# paths are less jagged and less tied to individual bin edges.
#
# Input data shape:
# - `z_grid`: 2D numpy array of normalized density values between 0 and 1
#
# Output data shape:
# - 2D numpy array of the same shape with a light weighted-neighbor blur applied
#   for contour rendering only; the underlying heatmap colors still use the
#   original thresholded grid

def smooth_grid_for_contours(z_grid: np.ndarray) -> np.ndarray:
    if z_grid.size == 0:
        return z_grid
    kernel = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    kernel /= kernel.sum()
    padded = np.pad(z_grid, ((1, 1), (1, 1)), mode='edge')
    smoothed = np.zeros_like(z_grid, dtype=float)
    for row_offset in range(3):
        for col_offset in range(3):
            smoothed += kernel[row_offset, col_offset] * padded[
                row_offset:row_offset + z_grid.shape[0],
                col_offset:col_offset + z_grid.shape[1],
            ]
    return smoothed


# Render one overlay panel for a specific dataset/bin-size/threshold
# combination.
#
# Input data shape:
# - `ax`: matplotlib axes for the panel
# - `spec_points`: list of `(HeatmapSpec, points)` tuples for one dataset after
#   outlier trimming
# - `bins`: integer bin count for the heatmap density grid
# - `dataset_label`: human-readable dataset label for panel annotation
# - `outlier_trim_percent`: fixed trim percentage used before binning
# - `threshold_mode`: active display-threshold mode
# - `threshold_value`: active threshold value for that mode
#
# Output data shape:
# - no return value
# - draws the overlay heatmap directly onto the provided axes
def render_panel(
    ax: plt.Axes,
    spec_points: list[tuple[HeatmapSpec, list[tuple[float, float, str]]]],
    bins: int,
    dataset_label: str,
    outlier_trim_percent: float,
    threshold_mode: str,
    threshold_value: float,
) -> None:
    ax.set_facecolor('white')
    ax.set_axisbelow(True)
    ax.grid(color='#e5e7eb', linewidth=0.55, alpha=0.22, zorder=0)
    threshold_label = THRESHOLD_MODE_LABELS[threshold_mode]
    contour_levels = build_contour_levels(threshold_mode, threshold_value)
    for spec, points in spec_points:
        color = TYPE_COLORS[spec.slug]
        x_centers, y_centers, z_grid = build_density_grid(points, bins=bins)
        thresholded_grid = apply_density_threshold(z_grid, threshold_mode, threshold_value)
        rgba = density_rgba(color, thresholded_grid, threshold_mode, threshold_value)
        extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
        ax.imshow(rgba, extent=extent, origin='lower', interpolation='bilinear', aspect='auto', zorder=2)
        class_max = float(thresholded_grid.max())
        if class_max > 0.0 and contour_levels:
            normalized_grid = thresholded_grid / class_max
            contour_grid = smooth_grid_for_contours(normalized_grid)
            ax.contour(
                x_centers,
                y_centers,
                contour_grid,
                levels=contour_levels,
                colors=[color] * len(contour_levels),
                linewidths=0.55,
                alpha=0.72,
                zorder=3,
            )
    ax.set_xlim(GLOBAL_X_RANGE[1], GLOBAL_X_RANGE[0])
    ax.set_ylim(GLOBAL_Y_RANGE[1], GLOBAL_Y_RANGE[0])
    ax.text(
        0.02,
        0.98,
        (
            f'{dataset_label}\n'
            f'bins={bins} | outlier_trim_percent={outlier_trim_percent:.2f}%\n'
            f'{threshold_label}={threshold_value:.3f}'
        ),
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=9,
        bbox={'facecolor': 'white', 'edgecolor': '#cbd5e1', 'alpha': 0.92, 'boxstyle': 'round,pad=0.3'},
        zorder=4,
    )

# Build one comparison-sheet figure for a dataset across many threshold values
# and bin sizes.
#
# Input data shape:
# - `dataset`: DatasetSpec with label and CSV path
# - `spec_points`: precomputed methyl overlay points after one grouped-data pass
# - `bins_list`: ordered list of integer bin sizes
# - `outlier_trim_percent`: fixed trim percentage applied before binning
# - `threshold_mode`: active display-threshold mode
# - `threshold_values`: ordered list of threshold values for that mode
#
# Output data shape:
# - a matplotlib Figure containing a threshold-by-bin grid of heatmap panels
#   plus a legend, ready to save as PDF or PNG
def render_dataset_figure(
    dataset: DatasetSpec,
    spec_points: list[tuple[HeatmapSpec, list[tuple[float, float, str]]]],
    bins_list: list[int],
    outlier_trim_percent: float,
    threshold_mode: str,
    threshold_values: list[float],
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(threshold_values),
        len(bins_list),
        figsize=(5.2 * len(bins_list), 4.3 * len(threshold_values)),
        squeeze=False,
    )
    for row_index, threshold_value in enumerate(threshold_values):
        for col_index, bins in enumerate(bins_list):
            ax = axes[row_index][col_index]
            render_panel(
                ax=ax,
                spec_points=spec_points,
                bins=bins,
                dataset_label=dataset.label,
                outlier_trim_percent=outlier_trim_percent,
                threshold_mode=threshold_mode,
                threshold_value=threshold_value,
            )
            ax.set_xlabel('Proton Chemical Shift (ppm)', fontsize=10)
            if col_index == len(bins_list) - 1:
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
                ax.set_ylabel('Carbon Chemical Shift (ppm)', fontsize=10)
            elif col_index == 0:
                ax.tick_params(axis='y', labelleft=True, labelright=False)
            else:
                ax.tick_params(axis='y', labelleft=False, labelright=False)
    handles = [
        Line2D([0], [0], color=TYPE_COLORS[spec.slug], linewidth=8, alpha=0.8, label=spec.title)
        for spec in SPECS
    ]
    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
        fontsize=10,
    )
    fig.suptitle(
        f'Methyl heatmap threshold sweep: {dataset.label}',
        fontsize=18,
        fontfamily='DejaVu Serif',
        y=0.968,
    )
    fig.subplots_adjust(top=0.82, left=0.09, right=0.93, bottom=0.10, hspace=0.14, wspace=0.08)
    return fig


# Write all requested exploratory comparison sheets to disk.
#
# Input data shape:
# - `datasets`: list of DatasetSpec objects, usually filtered and unfiltered
# - `bins_list`: ordered list of integer bin sizes
# - `outlier_trim_percent`: fixed trim percentage applied before binning
# - `threshold_mode`: active display-threshold mode
# - `threshold_values`: ordered list of threshold values for that mode
# - `output_dir`: directory where PDFs and optional PNGs should be written
# - `write_png`: boolean flag controlling whether PNG copies are also exported
#
# Output data shape:
# - no return value
# - writes one PDF per dataset, plus optional PNG files, into the output dir
def write_dataset_outputs(
    datasets: list[DatasetSpec],
    bins_list: list[int],
    outlier_trim_percent: float,
    threshold_mode: str,
    threshold_values: list[float],
    output_dir: Path,
    write_png: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_figures: list[tuple[DatasetSpec, plt.Figure]] = []

    for dataset in datasets:
        rows = read_shift_rows(dataset.csv_path)
        grouped = build_residue_atom_values(rows)
        spec_points = build_spec_points_from_grouped(grouped, outlier_trim_percent)
        figure = render_dataset_figure(
            dataset=dataset,
            spec_points=spec_points,
            bins_list=bins_list,
            outlier_trim_percent=outlier_trim_percent,
            threshold_mode=threshold_mode,
            threshold_values=threshold_values,
        )
        dataset_figures.append((dataset, figure))

    threshold_slug = threshold_mode.replace('-', '_')
    for dataset, figure in dataset_figures:
        pdf_path = output_dir / f'{dataset.slug}_methyl_heatmap_{threshold_slug}_sweep.pdf'
        figure.savefig(pdf_path, format='pdf', facecolor='white', bbox_inches='tight', pad_inches=0.18)
        print(f'Wrote {pdf_path}')
        if write_png:
            png_path = output_dir / f'{dataset.slug}_methyl_heatmap_{threshold_slug}_sweep.png'
            figure.savefig(png_path, format='png', dpi=220, facecolor='white', bbox_inches='tight', pad_inches=0.18)
            print(f'Wrote {png_path}')

    combined_pdf = output_dir / f'methyl_heatmap_{threshold_slug}_sweep_all_datasets.pdf'
    with PdfPages(combined_pdf) as pdf_pages:
        for _, figure in dataset_figures:
            pdf_pages.savefig(figure, facecolor='white', bbox_inches='tight', pad_inches=0.18)
    print(f'Wrote {combined_pdf}')

    for _, figure in dataset_figures:
        plt.close(figure)


# Run the exploratory methyl heatmap sweep from CLI arguments through output
# writing.
#
# Input data shape:
# - no direct function arguments; reads the script CLI and local CSV files
#
# Output data shape:
# - writes exploratory comparison-sheet files to disk
# - prints the locations of the generated outputs
# - raises exceptions if required CSV inputs are missing or invalid
def main() -> int:
    args = parse_args()
    ensure_file_exists(args.unfiltered_csv, 'Unfiltered assignment-level CSV')
    ensure_file_exists(args.filtered_csv, 'Filtered assignment-level CSV')
    datasets = [
        DatasetSpec('unfiltered', 'Unfiltered assignment-level dataset', args.unfiltered_csv),
        DatasetSpec('filtered', 'Filtered assignment-level dataset', args.filtered_csv),
    ]
    bins_list = sorted(dict.fromkeys(args.bins))
    threshold_values = sorted(dict.fromkeys(args.threshold_values))
    write_dataset_outputs(
        datasets=datasets,
        bins_list=bins_list,
        outlier_trim_percent=args.outlier_trim_percent,
        threshold_mode=args.threshold_mode,
        threshold_values=threshold_values,
        output_dir=args.output_dir,
        write_png=args.write_png,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
