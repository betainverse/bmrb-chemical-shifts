#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE_DIR = PROJECT_ROOT / "bmrb_data_cache"
CHARTS_DIR = PROJECT_ROOT / "carbon_proton_nitrogen_charts"
OUTPUT_DIR = CHARTS_DIR / "output"


AA_ORDER = [
    ("ALA", "A"),
    ("ARG", "R"),
    ("ASN", "N"),
    ("ASP", "D"),
    ("CYS", "C"),
    ("GLN", "Q"),
    ("GLU", "E"),
    ("GLY", "G"),
    ("HIS", "H"),
    ("ILE", "I"),
    ("LEU", "L"),
    ("LYS", "K"),
    ("MET", "M"),
    ("PHE", "F"),
    ("PRO", "P"),
    ("SER", "S"),
    ("THR", "T"),
    ("TRP", "W"),
    ("TYR", "Y"),
    ("VAL", "V"),
]

FILLED_COLORS = {
    "ND": "#60a5fa",
    "NE2_SIDE": "#f472b6",
    "NE1_SIDE": "#34d399",
    "NH": "#a78bfa",
    "NZ": "#f59e0b",
}

SS_STYLES = {
    "N": {"helix_fill": "#374151", "sheet_fill": "#9ca3af", "coil_outline": "#111827"},
}

AXIS_COLOR = "#374151"
LABEL_COLOR = "#111827"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": AXIS_COLOR,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,
        "axes.labelcolor": AXIS_COLOR,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sidechain_amide_nitrogen_bucket(atom: str, residue: str) -> str | None:
    atom = atom.upper()
    residue = residue.upper()
    if residue == "ASN" and atom == "ND2":
        return "ND"
    if residue == "GLN" and atom == "NE2":
        return "NE2_SIDE"
    if residue == "TRP" and atom == "NE1":
        return "NE1_SIDE"
    if residue == "HIS" and atom in {"ND1", "NE2"}:
        return "NH"
    return None


def sidechain_basic_nitrogen_bucket(atom: str, residue: str) -> str | None:
    atom = atom.upper()
    residue = residue.upper()
    if residue == "ARG" and atom == "NE":
        return "NE1_SIDE"
    if residue == "ARG" and atom in {"NH1", "NH2"}:
        return "NH"
    if residue == "LYS" and atom == "NZ":
        return "NZ"
    return None


def histidine_nitrogen_bucket(atom: str, residue: str) -> str | None:
    atom = atom.upper()
    residue = residue.upper()
    if residue != "HIS":
        return None
    if atom == "ND1":
        return "ND"
    if atom == "NE2":
        return "NE2_SIDE"
    return None


def read_filtered_compact(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                shifts = json.loads(row["shifts_json"])
            except Exception:
                continue
            rows.append(
                {
                    "residue_3": row["residue_3"],
                    "page_atom": row["page_atom"],
                    "trace_atom": row["trace_atom"],
                    "trace_name": row["trace_name"],
                    "shifts": [float(v) for v in shifts],
                }
            )
    return rows


def read_ss_compact(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                shifts = json.loads(row["shifts_json"])
            except Exception:
                continue
            rows.append(
                {
                    "residue_3": row["residue_3"],
                    "page_atom": row["page_atom"],
                    "secondary_structure": row["secondary_structure"],
                    "secondary_structure_raw": row["secondary_structure_raw"],
                    "shifts": [float(v) for v in shifts],
                }
            )
    return rows


def build_backbone_n(ss_rows: list[dict[str, object]]) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in ss_rows:
        if row["page_atom"] != "N":
            continue
        ss = str(row["secondary_structure"])
        if ss not in {"helix", "sheet", "coil"}:
            continue
        values[(str(row["residue_3"]), ss)].extend(row["shifts"])  # type: ignore[arg-type]
    return values


def build_filtered_buckets(filtered_rows: list[dict[str, object]], bucket_func) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in filtered_rows:
        residue = str(row["residue_3"])
        atom = str(row["trace_atom"])
        bucket = bucket_func(atom, residue)
        if bucket is None:
            continue
        values[(residue, bucket)].extend(row["shifts"])  # type: ignore[arg-type]
    return values


def main_page_axis_height() -> float:
    return 0.78 / (len(AA_ORDER) + 0.18 * (len(AA_ORDER) - 1))


def draw_ss_hist(ax, values, residue: str, bins: np.ndarray) -> int:
    helix_vals = values.get((residue, "helix"), [])
    sheet_vals = values.get((residue, "sheet"), [])
    coil_vals = values.get((residue, "coil"), [])
    styles = SS_STYLES["N"]
    counts = []
    for vals in [helix_vals, sheet_vals, coil_vals]:
        hist_counts, _ = np.histogram(vals, bins=bins)
        counts.append(int(hist_counts.max()) if len(hist_counts) else 0)
    ymax = max(max(counts), 1)
    if helix_vals:
        ax.hist(
            helix_vals,
            bins=bins,
            histtype="stepfilled",
            color=styles["helix_fill"],
            edgecolor=styles["helix_fill"],
            alpha=0.36,
            linewidth=0.8,
        )
    if sheet_vals:
        ax.hist(
            sheet_vals,
            bins=bins,
            histtype="stepfilled",
            color=styles["sheet_fill"],
            edgecolor=styles["sheet_fill"],
            alpha=0.36,
            linewidth=0.8,
        )
    if coil_vals:
        ax.hist(
            coil_vals,
            bins=bins,
            histtype="step",
            color=styles["coil_outline"],
            linewidth=0.9,
            linestyle=":",
        )
    return ymax


def make_nitrogen_figure(filtered_rows: list[dict[str, object]], ss_rows: list[dict[str, object]], n_bin_count: int):
    backbone_n = build_backbone_n(ss_rows)
    amide_values = build_filtered_buckets(filtered_rows, sidechain_amide_nitrogen_bucket)
    basic_values = build_filtered_buckets(filtered_rows, sidechain_basic_nitrogen_bucket)
    fig, axes = plt.subplots(nrows=len(AA_ORDER), ncols=1, figsize=(8.5, 11), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.90, top=0.84, bottom=0.06, hspace=0.18)
    bin_step = 50.0 / max(n_bin_count, 1)
    n_bins = np.arange(20.0, 150.0 + bin_step, bin_step)
    for row_idx, (residue, one) in enumerate(AA_ORDER):
        ax = axes[row_idx]
        ax_nd = ax.twinx()
        ax_ne2 = ax.twinx()
        ax_ne1 = ax.twinx()
        ax_basic = ax.twinx()
        ax_argnh = ax.twinx()
        ymax = draw_ss_hist(ax, backbone_n, residue, n_bins)
        nd_max = 1
        ne2_max = 1
        ne1_max = 1
        basic_max = 1
        argnh_max = 1
        for key, twin in [("ND", ax_nd), ("NE2_SIDE", ax_ne2), ("NE1_SIDE", ax_ne1)]:
            vals = [v for v in amide_values.get((residue, key), []) if 20.0 <= v <= 150.0]
            counts, _ = np.histogram(vals, bins=n_bins)
            local_max = max(int(counts.max()) if len(counts) else 0, 1)
            if key == "ND":
                nd_max = local_max
            elif key == "NE2_SIDE":
                ne2_max = local_max
            else:
                ne1_max = local_max
            if vals:
                twin.hist(vals, bins=n_bins, histtype="stepfilled", color=FILLED_COLORS[key], edgecolor=FILLED_COLORS[key], alpha=0.42, linewidth=0.8)
        for key in ["NE1_SIDE", "NZ"]:
            vals = [v for v in basic_values.get((residue, key), []) if 20.0 <= v < 100.0]
            counts, _ = np.histogram(vals, bins=n_bins)
            basic_max = max(basic_max, int(counts.max()) if len(counts) else 0, 1)
            if vals:
                ax_basic.hist(vals, bins=n_bins, histtype="stepfilled", color=FILLED_COLORS[key], edgecolor=FILLED_COLORS[key], alpha=0.40, linewidth=0.8)
        nh_vals = [v for v in basic_values.get((residue, "NH"), []) if 20.0 <= v < 100.0]
        nh_counts, _ = np.histogram(nh_vals, bins=n_bins)
        argnh_max = max(argnh_max, int(nh_counts.max()) if len(nh_counts) else 0, 1)
        if nh_vals:
            ax_argnh.hist(nh_vals, bins=n_bins, histtype="stepfilled", color=FILLED_COLORS["NH"], edgecolor=FILLED_COLORS["NH"], alpha=0.48, linewidth=0.8)
        ax.set_xlim(150, 20)
        ax.set_ylim(0, ymax * 1.05)
        ax_nd.set_ylim(0, nd_max * 1.05)
        ax_ne2.set_ylim(0, ne2_max * 1.05)
        ax_ne1.set_ylim(0, ne1_max * 1.05)
        ax_basic.set_ylim(0, basic_max * 1.05)
        ax_argnh.set_ylim(0, argnh_max * 1.05)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax_basic.tick_params(axis="y", right=False, labelright=False)
        ax_argnh.tick_params(axis="y", right=False, labelright=False)
        for twin in [ax_nd, ax_ne2, ax_ne1, ax_basic, ax_argnh]:
            twin.tick_params(axis="y", right=False, labelright=False)
            for side in ["right", "left", "top", "bottom"]:
                twin.spines[side].set_visible(False)
        ax.text(-0.05, 0.5, f"{residue} ({one})", transform=ax.transAxes, ha="right", va="center", fontsize=7.8, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
        if row_idx != len(AA_ORDER) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.tick_params(axis="x", labelsize=7)
    axes[0].set_title("", pad=0)
    axes[0].tick_params(axis="x", top=True, labeltop=True, labelsize=7, pad=1)
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(10.0))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax.tick_params(axis="x", which="minor", length=2, width=0.5)
    handles = [
        plt.Line2D([0], [0], color=SS_STYLES["N"]["helix_fill"], linewidth=6, alpha=0.6, label="N alpha"),
        plt.Line2D([0], [0], color=SS_STYLES["N"]["sheet_fill"], linewidth=6, alpha=0.6, label="N beta"),
        plt.Line2D([0], [0], color=SS_STYLES["N"]["coil_outline"], linewidth=0.9, linestyle=":", label="N coil"),
        plt.Line2D([0], [0], color=FILLED_COLORS["ND"], linewidth=6, alpha=0.42, label="ASN ND2"),
        plt.Line2D([0], [0], color=FILLED_COLORS["NE2_SIDE"], linewidth=6, alpha=0.42, label="GLN NE2"),
        plt.Line2D([0], [0], color=FILLED_COLORS["NE1_SIDE"], linewidth=6, alpha=0.40, label="TRP NE1 / ARG NE"),
        plt.Line2D([0], [0], color=FILLED_COLORS["NH"], linewidth=6, alpha=0.48, label="ARG NH1/NH2"),
        plt.Line2D([0], [0], color=FILLED_COLORS["NZ"], linewidth=6, alpha=0.40, label="LYS NZ"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.948), fontsize=8)
    fig.suptitle("Backbone and side-chain nitrogen chemical shifts from BMRB", fontsize=12.5, y=0.975, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
    fig.supxlabel("Nitrogen Chemical Shift (ppm)", fontsize=10, color=AXIS_COLOR)
    return fig


def make_histidine_nitrogen_figure(filtered_rows: list[dict[str, object]], n_bin_count: int):
    his_values = build_filtered_buckets(filtered_rows, histidine_nitrogen_bucket)
    fig = plt.figure(figsize=(8.5, 11))
    # Match the page-4 nitrogen chart scale:
    # main page width fraction = 0.79 for 130 ppm (150 -> 20)
    # histidine page width keeps the same ppm-per-inch over 50 ppm (200 -> 150)
    width = 0.79 * (50.0 / 130.0)
    height = main_page_axis_height()
    left = 0.35
    bottom = 0.82
    ax = fig.add_axes([left, bottom, width, height])
    ax_ne2 = ax.twinx()
    bins = np.arange(150.0, 200.0 + (50.0 / max(n_bin_count, 1)), 50.0 / max(n_bin_count, 1))

    nd_vals = [v for v in his_values.get(("HIS", "ND"), []) if 150.0 <= v <= 200.0]
    ne2_vals = [v for v in his_values.get(("HIS", "NE2_SIDE"), []) if 150.0 <= v <= 200.0]

    nd_counts, _ = np.histogram(nd_vals, bins=bins)
    ne2_counts, _ = np.histogram(ne2_vals, bins=bins)
    nd_max = max(int(nd_counts.max()) if len(nd_counts) else 0, 1)
    ne2_max = max(int(ne2_counts.max()) if len(ne2_counts) else 0, 1)

    if nd_vals:
        ax.hist(
            nd_vals,
            bins=bins,
            histtype="stepfilled",
            color=FILLED_COLORS["ND"],
            edgecolor=FILLED_COLORS["ND"],
            alpha=0.45,
            linewidth=0.8,
        )
    if ne2_vals:
        ax_ne2.hist(
            ne2_vals,
            bins=bins,
            histtype="stepfilled",
            color=FILLED_COLORS["NE2_SIDE"],
            edgecolor=FILLED_COLORS["NE2_SIDE"],
            alpha=0.45,
            linewidth=0.8,
        )

    ax.set_xlim(200, 150)
    ax.set_ylim(0, nd_max * 1.05)
    ax_ne2.set_ylim(0, ne2_max * 1.05)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax_ne2.tick_params(axis="y", right=False, labelright=False)
    for side in ["right", "left", "top", "bottom"]:
        ax_ne2.spines[side].set_visible(False)
    ax.tick_params(axis="x", top=True, labeltop=True, labelsize=7, pad=1)
    ax.xaxis.set_major_locator(MultipleLocator(10.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax.set_xlabel("Nitrogen Chemical Shift (ppm)", fontsize=9, color=AXIS_COLOR, labelpad=4)
    ax.text(
        -0.12,
        0.5,
        "HIS ND1 / NE2",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=8.2,
        fontweight="semibold",
        fontfamily="DejaVu Serif",
        color=LABEL_COLOR,
    )

    handles = [
        plt.Line2D([0], [0], color=FILLED_COLORS["ND"], linewidth=6, alpha=0.45, label="HIS ND1"),
        plt.Line2D([0], [0], color=FILLED_COLORS["NE2_SIDE"], linewidth=6, alpha=0.45, label="HIS NE2"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=8)
    fig.suptitle("Histidine side chain chemical shifts from BMRB", fontsize=12.5, y=0.965, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
    return fig


def save_figure(fig, pdf_path: Path, svg_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the nitrogen page from cached filtered and SS compact CSVs.")
    parser.add_argument(
        "--filtered-csv",
        type=Path,
        default=DATA_CACHE_DIR / "filtered_histogram_cache_compact.csv",
    )
    parser.add_argument(
        "--ss-csv",
        type=Path,
        default=DATA_CACHE_DIR / "ss_histogram_cache_compact.csv",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=OUTPUT_DIR / "bmrb_histogram_nitrogen_page.pdf",
    )
    parser.add_argument("--n-bins", type=int, default=160)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    filtered_rows = read_filtered_compact(args.filtered_csv)
    ss_rows = read_ss_compact(args.ss_csv)
    main_fig = make_nitrogen_figure(filtered_rows, ss_rows, args.n_bins)
    his_fig = make_histidine_nitrogen_figure(filtered_rows, args.n_bins)

    main_pdf = args.output_pdf
    main_svg = args.output_pdf.with_suffix(".svg")
    his_pdf = args.output_pdf.with_name(f"{args.output_pdf.stem}_histidine_sidechain.pdf")
    his_svg = his_pdf.with_suffix(".svg")

    save_figure(main_fig, main_pdf, main_svg)
    save_figure(his_fig, his_pdf, his_svg)
    plt.close(main_fig)
    plt.close(his_fig)
    print(f"Wrote {main_pdf}")
    print(f"Wrote {main_svg}")
    print(f"Wrote {his_pdf}")
    print(f"Wrote {his_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
