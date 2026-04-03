#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_DIR = PROJECT_ROOT / "methyl_heatmap_overlay"
DEFAULT_SHIFT_TABLE_CSV = (
    PROJECT_ROOT / "bmrb_data_cache" / "unfiltered_assignment_level_dataset" / "bmrb_shift_table_full.csv"
)

from render_methyl_heatmaps_html import (
    GLOBAL_X_RANGE,
    GLOBAL_Y_RANGE,
    SPECS,
    TYPE_COLORS,
    build_density_grid,
    build_residue_atom_values,
    extract_points,
    read_shift_rows,
    trim_outliers,
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


def build_spec_points(csv_path: Path) -> list[tuple[object, list[tuple[float, float, str]]]]:
    rows = read_shift_rows(csv_path)
    grouped = build_residue_atom_values(rows)
    spec_points = [(spec, trim_outliers(extract_points(grouped, spec))) for spec in SPECS]
    missing = [spec.title for spec, points in spec_points if not points]
    if missing:
        raise RuntimeError(f"No paired points found for: {', '.join(missing)}")
    return spec_points


def render_static_overlay(spec_points: list[tuple[object, list[tuple[float, float, str]]]]):
    fig = plt.figure(figsize=(11.0, 8.5), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[5.2, 1.0],
        height_ratios=[1.0, 5.2],
        left=0.10,
        right=0.92,
        top=0.82,
        bottom=0.12,
        wspace=0.05,
        hspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")

    for spec, points in spec_points:
        color = TYPE_COLORS[spec.slug]
        xs = np.array([x for x, _, _ in points], dtype=float)
        ys = np.array([y for _, y, _ in points], dtype=float)
        x_centers, y_centers, z_grid = build_density_grid(points, bins=220)
        rgba = density_rgba(color, z_grid)
        extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
        ax_main.imshow(
            rgba,
            extent=extent,
            origin="lower",
            interpolation="bilinear",
            aspect="auto",
            zorder=1,
        )
        ax_top.hist(xs, bins=120, range=GLOBAL_X_RANGE, color=color, alpha=0.42, histtype="stepfilled", linewidth=0)
        ax_right.hist(
            ys,
            bins=120,
            range=GLOBAL_Y_RANGE,
            orientation="horizontal",
            color=color,
            alpha=0.42,
            histtype="stepfilled",
            linewidth=0,
        )

    ax_main.set_xlim(GLOBAL_X_RANGE[1], GLOBAL_X_RANGE[0])
    ax_main.set_ylim(GLOBAL_Y_RANGE[1], GLOBAL_Y_RANGE[0])
    ax_main.set_xlabel("Methyl proton chemical shift (ppm)", fontsize=12)
    ax_main.set_ylabel("Methyl carbon chemical shift (ppm)", fontsize=12)
    ax_main.grid(color="#e5e7eb", linewidth=0.8)
    ax_main.set_facecolor("white")

    ax_top.set_xlim(GLOBAL_X_RANGE[1], GLOBAL_X_RANGE[0])
    ax_top.set_facecolor("white")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)

    ax_right.set_ylim(GLOBAL_Y_RANGE[1], GLOBAL_Y_RANGE[0])
    ax_right.set_facecolor("white")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=TYPE_COLORS[spec.slug], linewidth=8, alpha=0.8, label=spec.title)
        for spec, _ in spec_points
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
        fontsize=10,
    )
    fig.suptitle("Methyl Density Map", fontsize=18, fontfamily="DejaVu Serif", y=0.968)
    return fig


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


def main() -> int:
    args = build_parser().parse_args()
    spec_points = build_spec_points(args.csv)
    fig = render_static_overlay(spec_points)
    pdf_path = args.base_output.with_suffix(".pdf")
    svg_path = args.base_output.with_suffix(".svg")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    fig.savefig(svg_path, format="svg", facecolor="white")
    if args.write_png:
        png_path = args.base_output.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=220, facecolor="white")
        print(f"Wrote {png_path}")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
