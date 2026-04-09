#!/usr/bin/env python3
"""
Created: 2026-04-08

Purpose:
- render the production static methyl proton-carbon overlay heatmap from the
  assignment-level BMRB chemical shift CSV
- provide the default non-exploratory static figure used by the
  `methyl_heatmap_overlay/` workflow
- incorporate the visual improvements validated in the exploratory heatmap
  study, including the 120-bin density look, the default per-class
  relative-density threshold of 0.02, right-side carbon axis labeling, fainter
  grid lines, and smoothed thin contour lines
- keep a clearly documented `min-count = 5` alternative in the code for manual
  comparison, while leaving the production default on relative-density

Expected input data:
- `bmrb_shift_table_full.csv` in
  `chem_shifts/bmrb_data_cache/unfiltered_assignment_level_dataset/` by default
- the CSV is expected to contain assignment-level rows with residue identity,
  atom names, and numeric `shift_ppm` values

Expected output data:
- `bmrb_methyl_heatmaps_static.pdf`
- `bmrb_methyl_heatmaps_static.svg`
- optional `bmrb_methyl_heatmaps_static.png`
- all outputs are written under `chem_shifts/methyl_heatmap_overlay/` unless a
  different `--base-output` is provided

Relationship to other workflows:
- this is the production companion to the exploratory script in
  `chem_shifts/methyl_heatmap_exploration/`
- it is intentionally standalone and does not depend on the HTML renderer
- unlike the exploratory script, it renders one polished default figure rather
  than a parameter sweep

Command-line use:
- default invocation:
  `python3 methyl_heatmap_overlay/render_methyl_heatmaps_static.py`
- common options:
  `--csv` to point at a different assignment-level shift CSV
  `--base-output` to choose a different output basename
  `--write-png` to also export a PNG preview
- example:
  `python3 methyl_heatmap_overlay/render_methyl_heatmaps_static.py --write-png`
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_DIR = PROJECT_ROOT / "methyl_heatmap_overlay"
DEFAULT_SHIFT_TABLE_CSV = (
    PROJECT_ROOT / "bmrb_data_cache" / "unfiltered_assignment_level_dataset" / "bmrb_shift_table_full.csv"
)
DEFAULT_HEATMAP_BINS = 120
DEFAULT_HISTOGRAM_BINS = 120
DEFAULT_THRESHOLD_MODE = "relative-density"
DEFAULT_THRESHOLD_VALUE = 0.02
# Manual alternative:
# - set `DEFAULT_THRESHOLD_MODE = "min-count"`
# - set `DEFAULT_THRESHOLD_VALUE = 5.0`
# if you want the stronger minimum-count look that was the runner-up in the
# exploratory comparisons.
DEFAULT_CONTOUR_LEVELS = [0.15, 0.30, 0.50, 0.70]
GLOBAL_X_RANGE = [-0.5, 3.0]
GLOBAL_Y_RANGE = [8.0, 30.0]

TYPE_COLORS = {
    "ala-hb-cb": "#e11d48",
    "ile-hd1-cd1": "#0f766e",
    "ile-hg2-cg2": "#65a30d",
    "leu-hd-cd": "#8b5cf6",
    "val-hg-cg": "#2563eb",
    "met-he-ce": "#c026d3",
    "thr-hg2-cg2": "#c2410c",
}

RESIDUE_KEY_FIELDS = (
    "bmrb_id",
    "entity_id",
    "entity_assembly_id",
    "comp_index_id",
    "assigned_chem_shift_list_id",
    "residue_3",
)


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
    x_label: str
    y_label: str
    variants: tuple[PairVariant, ...]


SPECS = (
    HeatmapSpec(
        slug="ala-hb-cb",
        title="Alanine HB/CB",
        residue="ALA",
        x_label="HB / MB (ppm)",
        y_label="CB (ppm)",
        variants=(PairVariant("ALA HB-CB", ("HB1", "HB2", "HB3"), ("CB",)),),
    ),
    HeatmapSpec(
        slug="ile-hd1-cd1",
        title="Isoleucine HD1/CD1",
        residue="ILE",
        x_label="HD1 (ppm)",
        y_label="CD1 (ppm)",
        variants=(PairVariant("ILE HD1-CD1", ("HD11", "HD12", "HD13"), ("CD1",)),),
    ),
    HeatmapSpec(
        slug="ile-hg2-cg2",
        title="Isoleucine HG2/CG2",
        residue="ILE",
        x_label="HG2 (ppm)",
        y_label="CG2 (ppm)",
        variants=(PairVariant("ILE HG2-CG2", ("HG21", "HG22", "HG23"), ("CG2",)),),
    ),
    HeatmapSpec(
        slug="val-hg-cg",
        title="Valine HG/CG",
        residue="VAL",
        x_label="HG (ppm)",
        y_label="CG (ppm)",
        variants=(
            PairVariant("VAL HG1-CG1", ("HG11", "HG12", "HG13"), ("CG1",)),
            PairVariant("VAL HG2-CG2", ("HG21", "HG22", "HG23"), ("CG2",)),
        ),
    ),
    HeatmapSpec(
        slug="leu-hd-cd",
        title="Leucine HD/CD",
        residue="LEU",
        x_label="HD (ppm)",
        y_label="CD (ppm)",
        variants=(
            PairVariant("LEU HD1-CD1", ("HD11", "HD12", "HD13"), ("CD1",)),
            PairVariant("LEU HD2-CD2", ("HD21", "HD22", "HD23"), ("CD2",)),
        ),
    ),
    HeatmapSpec(
        slug="met-he-ce",
        title="Methionine HE/CE",
        residue="MET",
        x_label="HE / ME (ppm)",
        y_label="CE (ppm)",
        variants=(PairVariant("MET HE-CE", ("HE1", "HE2", "HE3"), ("CE",)),),
    ),
    HeatmapSpec(
        slug="thr-hg2-cg2",
        title="Threonine HG2/CG2",
        residue="THR",
        x_label="HG2 (ppm)",
        y_label="CG2 (ppm)",
        variants=(PairVariant("THR HG2-CG2", ("HG21", "HG22", "HG23"), ("CG2",)),),
    ),
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#0f172a",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
    }
)


# Read the assignment-level shift CSV into row dictionaries for local
# methyl-heatmap processing.
#
# Input data shape:
# - `path`: CSV file containing assignment-level BMRB chemical shift rows
#
# Output data shape:
# - list of `dict[str, str]`, one dictionary per CSV row
def read_shift_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# Group assignment-level shift rows by residue instance and atom name so methyl
# proton/carbon pairs can be reconstructed.
#
# Input data shape:
# - `rows`: assignment-level CSV rows with residue identity, atom names, and
#   numeric `shift_ppm` values stored as text
#
# Output data shape:
# - nested dictionary keyed first by residue identity tuple and then by atom
#   name, where each atom entry stores a list of float shift values
def build_residue_atom_values(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, list[float]]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        atom = row["atom"].upper()
        residue = row["residue_3"].upper()
        if residue not in {"ALA", "ILE", "LEU", "VAL", "MET", "THR"}:
            continue
        if not atom.startswith(("H", "C")):
            continue
        try:
            shift = float(row["shift_ppm"])
        except ValueError:
            continue
        key = tuple(row[field] for field in RESIDUE_KEY_FIELDS)
        grouped[key][atom].append(shift)
    return grouped


# Convert grouped residue-level atom values into paired methyl proton/carbon
# points for one heatmap specification.
#
# Input data shape:
# - `grouped`: residue-level nested atom-value dictionary
# - `spec`: one HeatmapSpec describing which residue and atom variants to pair
#
# Output data shape:
# - list of `(proton_shift_ppm, carbon_shift_ppm, variant_label)` tuples
def extract_points(
    grouped: dict[tuple[str, ...], dict[str, list[float]]],
    spec: HeatmapSpec,
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


# Remove extreme paired points symmetrically from both axes before heatmap
# binning so a small number of outliers do not dominate the display.
#
# Input data shape:
# - `points`: list of paired methyl proton/carbon tuples
# - `percentile`: percentage to trim from each tail of both axes
#
# Output data shape:
# - filtered list of paired methyl proton/carbon tuples
def trim_outliers(points: list[tuple[float, float, str]], percentile: float = 0.5) -> list[tuple[float, float, str]]:
    xs = np.array([x for x, _, _ in points], dtype=float)
    ys = np.array([y for _, y, _ in points], dtype=float)
    x_low, x_high = np.percentile(xs, [percentile, 100.0 - percentile])
    y_low, y_high = np.percentile(ys, [percentile, 100.0 - percentile])
    return [(x, y, label) for x, y, label in points if x_low <= x <= x_high and y_low <= y <= y_high]


# Bin paired methyl proton/carbon points into a 2D histogram grid for heatmap
# rendering.
#
# Input data shape:
# - `points`: list of paired methyl proton/carbon tuples
# - `bins`: integer bin count applied along both axes
#
# Output data shape:
# - tuple of x centers, y centers, and transposed 2D histogram counts
def build_density_grid(points: list[tuple[float, float, str]], bins: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array([x for x, _, _ in points], dtype=float)
    ys = np.array([y for _, y, _ in points], dtype=float)
    x_edges = np.linspace(GLOBAL_X_RANGE[0], GLOBAL_X_RANGE[1], bins + 1)
    y_edges = np.linspace(GLOBAL_Y_RANGE[0], GLOBAL_Y_RANGE[1], bins + 1)
    hist, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    return x_centers, y_centers, hist.T


# Convert one methyl-class density grid into a tinted RGBA image for imshow.
#
# Input data shape:
# - `hex_color`: matplotlib-compatible hex color for one methyl class
# - `z_grid`: 2D numpy array of histogram counts for one methyl class
#
# Output data shape:
# - `(height, width, 4)` numpy array with RGB color channels and an alpha
#   channel that increases with local density
def density_rgba(hex_color: str, z_grid: np.ndarray) -> np.ndarray:
    rgb = np.array(to_rgb(hex_color), dtype=float)
    if float(z_grid.max()) <= 0:
        norm = np.zeros_like(z_grid, dtype=float)
    else:
        norm = z_grid / float(z_grid.max())
    alpha = np.clip(norm ** 0.55, 0.0, 1.0) * 0.90
    rgba = np.zeros(z_grid.shape + (4,), dtype=float)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return rgba


# Apply the selected production threshold to one methyl-class density grid
# before it is converted into color and contour layers.
#
# Input data shape:
# - `z_grid`: 2D numpy array of histogram counts for one methyl class
# - `threshold_mode`: either `relative-density` or `min-count`
# - `threshold_value`: numeric threshold for the chosen mode
#
# Output data shape:
# - 2D numpy array of histogram counts with low-density or low-count bins
#   zeroed out according to the selected mode
def apply_density_threshold(
    z_grid: np.ndarray,
    threshold_mode: str,
    threshold_value: float,
) -> np.ndarray:
    filtered = np.array(z_grid, copy=True)
    if threshold_mode == "min-count":
        filtered[filtered < threshold_value] = 0.0
        return filtered
    class_max = float(filtered.max())
    if class_max <= 0.0:
        return filtered
    filtered[filtered < (class_max * threshold_value)] = 0.0
    return filtered


# Build the contour levels that should remain visible after thresholding.
#
# Input data shape:
# - `threshold_mode`: either `relative-density` or `min-count`
# - `threshold_value`: numeric threshold for the chosen mode
#
# Output data shape:
# - list of normalized contour levels between 0 and 1
def build_contour_levels(threshold_mode: str, threshold_value: float) -> list[float]:
    if threshold_mode == "relative-density":
        return [level for level in DEFAULT_CONTOUR_LEVELS if level > threshold_value]
    return list(DEFAULT_CONTOUR_LEVELS)


# Smooth a normalized density grid before contour extraction so contour paths
# are less jagged and less tied to individual bin edges.
#
# Input data shape:
# - `z_grid`: 2D numpy array of normalized density values between 0 and 1
#
# Output data shape:
# - 2D numpy array of the same shape with a light weighted-neighbor blur
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
    padded = np.pad(z_grid, ((1, 1), (1, 1)), mode="edge")
    smoothed = np.zeros_like(z_grid, dtype=float)
    for row_offset in range(3):
        for col_offset in range(3):
            smoothed += kernel[row_offset, col_offset] * padded[
                row_offset:row_offset + z_grid.shape[0],
                col_offset:col_offset + z_grid.shape[1],
            ]
    return smoothed


# Read the assignment-level shift table and build trimmed methyl-pair points for
# each production heatmap specification.
#
# Input data shape:
# - `csv_path`: path to an assignment-level shift CSV with residue, atom, and
#   `shift_ppm` columns
#
# Output data shape:
# - list of `(spec, points)` tuples where each point is
#   `(proton_shift_ppm, carbon_shift_ppm, variant_label)`
def build_spec_points(csv_path: Path) -> list[tuple[object, list[tuple[float, float, str]]]]:
    rows = read_shift_rows(csv_path)
    grouped = build_residue_atom_values(rows)
    spec_points = [(spec, trim_outliers(extract_points(grouped, spec))) for spec in SPECS]
    missing = [spec.title for spec, points in spec_points if not points]
    if missing:
        raise RuntimeError(f"No paired points found for: {', '.join(missing)}")
    return spec_points


# Render the production static methyl overlay figure with marginal histograms.
#
# Input data shape:
# - `spec_points`: list of `(spec, points)` tuples where points are paired
#   methyl proton/carbon coordinates for each methyl class
#
# Output data shape:
# - matplotlib Figure containing the main overlay panel plus top/right marginal
#   histograms and the shared legend/title
def render_static_overlay(spec_points: list[tuple[object, list[tuple[float, float, str]]]]):
    fig = plt.figure(figsize=(11.0, 8.5), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 5.2],
        height_ratios=[1.0, 5.2],
        left=0.10,
        right=0.92,
        top=0.82,
        bottom=0.12,
        wspace=0.05,
        hspace=0.05,
    )
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1], sharex=ax_top, sharey=ax_left)
    ax_corner.axis("off")

    ax_main.set_facecolor("white")
    ax_main.set_axisbelow(True)
    ax_main.grid(color="#e5e7eb", linewidth=0.55, alpha=0.22, zorder=0)
    contour_levels = build_contour_levels(DEFAULT_THRESHOLD_MODE, DEFAULT_THRESHOLD_VALUE)

    for spec, points in spec_points:
        color = TYPE_COLORS[spec.slug]
        xs = np.array([x for x, _, _ in points], dtype=float)
        ys = np.array([y for _, y, _ in points], dtype=float)
        x_centers, y_centers, z_grid = build_density_grid(points, bins=DEFAULT_HEATMAP_BINS)
        thresholded_grid = apply_density_threshold(z_grid, DEFAULT_THRESHOLD_MODE, DEFAULT_THRESHOLD_VALUE)
        rgba = density_rgba(color, thresholded_grid)
        extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
        ax_main.imshow(
            rgba,
            extent=extent,
            origin="lower",
            interpolation="bilinear",
            aspect="auto",
            zorder=2,
        )
        class_max = float(thresholded_grid.max())
        if class_max > 0.0 and contour_levels:
            normalized_grid = thresholded_grid / class_max
            contour_grid = smooth_grid_for_contours(normalized_grid)
            ax_main.contour(
                x_centers,
                y_centers,
                contour_grid,
                levels=contour_levels,
                colors=[color] * len(contour_levels),
                linewidths=0.55,
                alpha=0.72,
                zorder=3,
            )
        ax_top.hist(
            xs,
            bins=DEFAULT_HISTOGRAM_BINS,
            range=GLOBAL_X_RANGE,
            color=color,
            alpha=0.42,
            histtype="stepfilled",
            linewidth=0,
        )
        ax_left.hist(
            ys,
            bins=DEFAULT_HISTOGRAM_BINS,
            range=GLOBAL_Y_RANGE,
            orientation="horizontal",
            color=color,
            alpha=0.42,
            histtype="stepfilled",
            linewidth=0,
        )

    ax_main.set_xlim(GLOBAL_X_RANGE[1], GLOBAL_X_RANGE[0])
    ax_main.set_ylim(GLOBAL_Y_RANGE[1], GLOBAL_Y_RANGE[0])
    ax_main.set_xlabel("Methyl Proton Chemical Shift (ppm)", fontsize=12)
    ax_main.yaxis.set_label_position("right")
    ax_main.yaxis.tick_right()
    ax_main.set_ylabel("Methyl Carbon Chemical Shift (ppm)", fontsize=12)

    ax_top.set_xlim(GLOBAL_X_RANGE[1], GLOBAL_X_RANGE[0])
    ax_top.set_facecolor("white")
    ax_top.xaxis.set_ticks_position("bottom")
    ax_top.tick_params(axis="x", bottom=True, top=False, labelbottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)

    ax_left.set_ylim(GLOBAL_Y_RANGE[1], GLOBAL_Y_RANGE[0])
    ax_left.set_facecolor("white")
    ax_left.tick_params(axis="y", labelleft=False, labelright=False)
    ax_left.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False, length=0)
    ax_left.set_xticks([])
    ax_left.spines["left"].set_visible(False)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["bottom"].set_visible(False)
    ax_left.invert_xaxis()

    handles = [
        Line2D([0], [0], color=TYPE_COLORS[spec.slug], linewidth=8, alpha=0.8, label=spec.title)
        for spec, _ in spec_points
    ]
    main_panel_box = ax_main.get_position()
    main_panel_center_x = (main_panel_box.x0 + main_panel_box.x1) / 2.0
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(main_panel_center_x, 0.93),
        fontsize=10,
    )
    fig.suptitle(
        "Methyl Density Map from BMRB data",
        fontsize=18,
        fontfamily="DejaVu Serif",
        x=main_panel_center_x,
        y=0.968,
    )
    return fig


# Define the command-line interface for the production static renderer.
#
# Input data shape:
# - no direct input values; reads command-line arguments from the shell
#
# Output data shape:
# - ArgumentParser configured with CSV and output-path options
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a static methyl density overlay to PDF/SVG.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_SHIFT_TABLE_CSV)
    parser.add_argument(
        "--base-output",
        type=Path,
        default=OVERLAY_DIR / "bmrb_methyl_heatmaps_static",
        help="Output base path without extension.",
    )
    parser.add_argument("--write-png", action="store_true", help="Also write a PNG preview.")
    return parser


# Run the production static methyl heatmap rendering workflow.
#
# Input data shape:
# - shell arguments parsed by `build_parser`
#
# Output data shape:
# - writes PDF/SVG outputs, and optionally PNG, to the requested base path
# - returns process exit code `0` on success
def main() -> int:
    args = build_parser().parse_args()
    spec_points = build_spec_points(args.csv)
    fig = render_static_overlay(spec_points)
    pdf_path = args.base_output.with_suffix(".pdf")
    svg_path = args.base_output.with_suffix(".svg")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", facecolor="white", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(svg_path, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.18)
    if args.write_png:
        png_path = args.base_output.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.18)
        print(f"Wrote {png_path}")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
