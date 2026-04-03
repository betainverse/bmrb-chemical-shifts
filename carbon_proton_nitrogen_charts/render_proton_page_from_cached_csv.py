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
    "HB": "#f472b6",
    "HG": "#34d399",
    "MG": "#86efac",
    "HD": "#a78bfa",
    "HE": "#f59e0b",
    "HE1": "#f59e0b",
    "HE2": "#f59e0b",
    "HZ": "#ef4444",
}

SS_STYLES = {
    "H": {"helix_fill": "#374151", "sheet_fill": "#9ca3af", "coil_outline": "#111827"},
    "HA": {"helix_fill": "#1d4ed8", "sheet_fill": "#60a5fa", "coil_outline": "#2563eb"},
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


def aliphatic_proton_bucket(atom: str, residue: str) -> str | None:
    atom = atom.upper()
    residue = residue.upper()
    if atom in {"H", "HN"}:
        return None
    if atom.startswith("HA"):
        return "HA"
    if residue == "ALA" and atom == "MB":
        return "HB"
    if atom.startswith("HB"):
        return "HB"
    if atom.startswith("HG"):
        if residue in {"PHE", "TYR", "TRP", "HIS"}:
            return None
        if residue == "ILE" and atom.startswith("HG1"):
            return "HG"
        return "HG"
    if residue == "VAL" and atom in {"MG1", "MG2"}:
        return "HG"
    if residue == "ILE" and atom == "MG":
        return "MG"
    if atom.startswith("HD"):
        if residue == "ASP" or residue in {"PHE", "TYR", "TRP", "HIS"}:
            return None
        return "HD"
    if residue == "LEU" and atom in {"MD1", "MD2"}:
        return "HD"
    if residue == "ILE" and atom == "MD":
        return "HD"
    if atom.startswith("HE"):
        if residue == "GLU" or residue in {"PHE", "TYR", "TRP", "HIS"}:
            return None
        return "HE"
    if residue == "MET" and atom == "ME":
        return "HE"
    if atom.startswith("HZ") or atom.startswith("HH"):
        if residue in {"PHE", "TYR", "TRP", "HIS"}:
            return None
        return "HE"
    return None


def aromatic_proton_bucket(atom: str, residue: str) -> str | None:
    atom = atom.upper()
    residue = residue.upper()
    if residue not in {"PHE", "TYR", "TRP", "HIS"}:
        return None
    if atom.startswith("HG"):
        return "HG"
    if atom.startswith("HD"):
        if residue == "HIS" and atom.startswith("HD1"):
            return None
        return "HD"
    if atom.startswith("HE"):
        if residue == "HIS" and atom.startswith("HE1"):
            return "HE1"
        if residue == "HIS" and atom.startswith("HE2"):
            return None
        if residue == "TRP" and atom.startswith("HE1"):
            return "HE1"
        if residue == "TRP" and atom.startswith("HE3"):
            return "HE2"
        return "HE"
    if atom.startswith("HZ") or atom.startswith("HH"):
        return "HZ"
    return None


def filtered_proton_bucket(row: dict[str, object]) -> str | None:
    residue = str(row["residue_3"]).upper()
    page_atom = str(row["page_atom"]).upper()
    trace_atom = str(row["trace_atom"]).upper()

    if residue == "ALA" and page_atom == "MB":
        return "HB"
    if residue == "MET" and page_atom == "ME":
        return "HE"
    if residue == "VAL" and page_atom in {"MG1", "MG2"}:
        return "HG"
    if residue == "LEU" and page_atom in {"MD1", "MD2"}:
        return "HD"
    if residue == "THR" and page_atom == "MG":
        return "MG"
    if residue == "ILE":
        if page_atom in {"HG12", "HG13"}:
            return "HG"
        if page_atom == "MG":
            if trace_atom.startswith("HG2"):
                return "MG"
            return None
        if page_atom == "MD":
            return "HD"

    bucket = aliphatic_proton_bucket(trace_atom, residue)
    if bucket is not None:
        return bucket
    return aromatic_proton_bucket(trace_atom, residue)


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


def build_backbone_ss(ss_rows: list[dict[str, object]], page_atom: str) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in ss_rows:
        if row["page_atom"] != page_atom:
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
        try:
            bucket = bucket_func(row)
        except TypeError:
            atom = str(row["trace_atom"])
            bucket = bucket_func(atom, residue)
        if bucket is None:
            continue
        values[(residue, bucket)].extend(row["shifts"])  # type: ignore[arg-type]
    return values


def draw_ss_hist(ax, values, residue: str, atom: str, bins: np.ndarray) -> int:
    helix_vals = values.get((residue, "helix"), [])
    sheet_vals = values.get((residue, "sheet"), [])
    coil_vals = values.get((residue, "coil"), [])
    styles = SS_STYLES[atom]
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


def make_proton_figure(filtered_rows: list[dict[str, object]], ss_rows: list[dict[str, object]], proton_bin_count: int):
    backbone_h = build_backbone_ss(ss_rows, "H")
    ha_ss = build_backbone_ss(ss_rows, "HA")
    proton_values = build_filtered_buckets(filtered_rows, filtered_proton_bucket)
    fig, axes = plt.subplots(nrows=len(AA_ORDER), ncols=1, figsize=(8.5, 11), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.90, top=0.84, bottom=0.06, hspace=0.18)
    proton_bins = np.linspace(-1.0, 12.0, proton_bin_count + 1)
    for row_idx, (residue, one) in enumerate(AA_ORDER):
        ax = axes[row_idx]
        proton_axes = {key: ax.twinx() for key in ["H", "HA", "HB", "HG", "MG", "HD", "HE", "HE1", "HE2", "HZ"]}
        maxima: dict[str, int] = {}
        maxima["H"] = draw_ss_hist(proton_axes["H"], backbone_h, residue, "H", proton_bins)
        maxima["HA"] = draw_ss_hist(proton_axes["HA"], ha_ss, residue, "HA", proton_bins)
        for key in ["HB", "HG", "MG", "HD", "HE", "HE1", "HE2"]:
            vals = list(proton_values.get((residue, key), []))
            counts, _ = np.histogram(vals, bins=proton_bins)
            maxima[key] = max(int(counts.max()) if len(counts) else 0, 1)
            if vals:
                proton_axes[key].hist(
                    vals,
                    bins=proton_bins,
                    histtype="stepfilled",
                    color=FILLED_COLORS[key],
                    edgecolor=FILLED_COLORS[key],
                    alpha=0.45 if key == "HB" else 0.38,
                    linewidth=0.8,
                )
        hz_vals = list(proton_values.get((residue, "HZ"), []))
        counts, _ = np.histogram(hz_vals, bins=proton_bins)
        maxima["HZ"] = max(int(counts.max()) if len(counts) else 0, 1)
        if hz_vals:
            proton_axes["HZ"].hist(
                hz_vals,
                bins=proton_bins,
                histtype="stepfilled",
                color=FILLED_COLORS["HZ"],
                edgecolor=FILLED_COLORS["HZ"],
                alpha=0.40,
                linewidth=0.8,
            )
        ax.set_xlim(12, -1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", left=False, labelleft=False)
        for key, twin in proton_axes.items():
            ymax = maxima[key]
            if (residue in {"CYS", "SER"} and key == "HG") or (residue == "THR" and key == "HG") or (residue == "TYR" and key == "HZ"):
                twin.set_ylim(0, max(ymax * 2.0, 1))
            else:
                twin.set_ylim(0, ymax * 1.05)
            twin.tick_params(axis="y", right=False, labelright=False)
            for side in ["right", "left", "top", "bottom"]:
                twin.spines[side].set_visible(False)
        if residue == "TRP":
            if maxima.get("HE1", 0) > 1:
                proton_axes["HE1"].annotate(
                    "HE1",
                    xy=(10.1, maxima["HE1"] * 0.72),
                    xytext=(10.7, maxima["HE1"] * 0.84),
                    fontsize=6.5,
                    color="#000000",
                    arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8},
                )
            if maxima.get("HE2", 0) > 1:
                proton_axes["HE2"].annotate(
                    "HE3",
                    xy=(7.2, maxima["HE2"] * 0.64),
                    xytext=(7.9, maxima["HE2"] * 0.80),
                    fontsize=6.5,
                    color="#000000",
                    arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8},
                )
        if residue == "ILE":
            if maxima.get("HG", 0) > 1:
                proton_axes["HG"].annotate(
                    "HG1",
                    xy=(1.35, maxima["HG"] * 0.70),
                    xytext=(1.95, maxima["HG"] * 0.84),
                    fontsize=6.5,
                    color="#000000",
                    arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8},
                )
            if maxima.get("MG", 0) > 1:
                proton_axes["MG"].annotate(
                    "HG2",
                    xy=(0.72, maxima["MG"] * 0.68),
                    xytext=(1.30, maxima["MG"] * 0.84),
                    fontsize=6.5,
                    color="#000000",
                    arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8},
                )
        ax.text(
            -0.05,
            0.5,
            f"{residue} ({one})",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=7.8,
            fontweight="semibold",
            fontfamily="DejaVu Serif",
            color=LABEL_COLOR,
        )
        if row_idx != len(AA_ORDER) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.tick_params(axis="x", labelsize=7)
    axes[0].set_title("", pad=0)
    axes[0].tick_params(axis="x", top=True, labeltop=True, labelsize=7, pad=1)
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="x", which="minor", length=2, width=0.5)
    handles = [
        plt.Line2D([0], [0], color=SS_STYLES["H"]["helix_fill"], linewidth=6, alpha=0.6, label="H alpha"),
        plt.Line2D([0], [0], color=SS_STYLES["H"]["sheet_fill"], linewidth=6, alpha=0.6, label="H beta"),
        plt.Line2D([0], [0], color=SS_STYLES["H"]["coil_outline"], linewidth=0.9, linestyle=":", label="H coil"),
        plt.Line2D([0], [0], color=SS_STYLES["HA"]["helix_fill"], linewidth=6, alpha=0.6, label="HA alpha"),
        plt.Line2D([0], [0], color=SS_STYLES["HA"]["sheet_fill"], linewidth=6, alpha=0.6, label="HA beta"),
        plt.Line2D([0], [0], color=SS_STYLES["HA"]["coil_outline"], linewidth=0.9, linestyle=":", label="HA coil"),
        plt.Line2D([0], [0], color=FILLED_COLORS["HB"], linewidth=6, alpha=0.45, label="HB"),
        plt.Line2D([0], [0], color=FILLED_COLORS["HG"], linewidth=6, alpha=0.38, label="HG"),
        plt.Line2D([0], [0], color=FILLED_COLORS["MG"], linewidth=6, alpha=0.38, label="MG"),
        plt.Line2D([0], [0], color=FILLED_COLORS["HD"], linewidth=6, alpha=0.38, label="HD"),
        plt.Line2D([0], [0], color=FILLED_COLORS["HE1"], linewidth=6, alpha=0.38, label="HE1"),
        plt.Line2D([0], [0], color=FILLED_COLORS["HE2"], linewidth=6, alpha=0.38, label="HE2/HE3"),
        plt.Line2D([0], [0], color=FILLED_COLORS["HZ"], linewidth=6, alpha=0.38, label="HZ / HH"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.948), fontsize=8)
    fig.suptitle("Backbone and side-chain proton chemical shifts from BMRB", fontsize=12.5, y=0.975, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
    fig.supxlabel("Hydrogen Chemical Shift (ppm)", fontsize=10, color=AXIS_COLOR)
    return fig


def make_histidine_proton_figure(filtered_rows: list[dict[str, object]], proton_bin_count: int):
    his_hd1_vals: list[float] = []
    his_he2_vals: list[float] = []
    for row in filtered_rows:
        if row["residue_3"] != "HIS":
            continue
        atom = str(row["trace_atom"]).upper()
        vals = [float(v) for v in row["shifts"]]  # type: ignore[arg-type]
        vals = [v for v in vals if 6.0 <= v <= 16.0]
        if atom == "HD1":
            his_hd1_vals.extend(vals)
        elif atom == "HE2":
            his_he2_vals.extend(vals)

    bins = np.linspace(6.0, 16.0, max(120, proton_bin_count) + 1)
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.35, 0.82, 0.30, 0.0305])
    ax_he2 = ax.twinx()

    hd1_counts, _ = np.histogram(his_hd1_vals, bins=bins)
    he2_counts_unweighted, _ = np.histogram(his_he2_vals, bins=bins)
    hd1_peak = int(hd1_counts.max()) if len(hd1_counts) else 0
    he2_peak = int(he2_counts_unweighted.max()) if len(he2_counts_unweighted) else 0
    he2_weight = 0.5 if he2_peak == 0 else min(1.0, 0.5 * hd1_peak / he2_peak) if hd1_peak else 0.5
    he2_weights = np.full(len(his_he2_vals), he2_weight) if his_he2_vals else None
    he2_counts, _ = np.histogram(his_he2_vals, bins=bins, weights=he2_weights)
    he2_ymax = max(int(he2_counts.max()) if len(he2_counts) else 0, hd1_peak, 1)

    if his_hd1_vals:
        ax.hist(
            his_hd1_vals,
            bins=bins,
            histtype="stepfilled",
            color=FILLED_COLORS["HD"],
            edgecolor=FILLED_COLORS["HD"],
            alpha=0.45,
            linewidth=0.8,
        )
    if his_he2_vals:
        ax_he2.hist(
            his_he2_vals,
            bins=bins,
            weights=he2_weights,
            histtype="stepfilled",
            color=FILLED_COLORS["HE2"],
            edgecolor=FILLED_COLORS["HE2"],
            alpha=0.45,
            linewidth=0.8,
        )

    ax.set_xlim(16, 6)
    ax.set_ylim(0, max(hd1_peak, 1) * 1.05)
    ax_he2.set_ylim(0, he2_ymax * 1.05)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax_he2.tick_params(axis="y", right=False, labelright=False)
    for side in ["right", "left", "top", "bottom"]:
        ax_he2.spines[side].set_visible(False)
    ax.tick_params(axis="x", top=True, labeltop=True, labelsize=7, pad=1)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax.set_xlabel("Hydrogen Chemical Shift (ppm)", fontsize=9, color=AXIS_COLOR, labelpad=4)
    ax.text(
        -0.14,
        0.5,
        "HIS HD1 / HE2",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=8.2,
        fontweight="semibold",
        fontfamily="DejaVu Serif",
        color=LABEL_COLOR,
    )

    handles = [
        plt.Line2D([0], [0], color=FILLED_COLORS["HD"], linewidth=6, alpha=0.45, label="HIS HD1"),
        plt.Line2D([0], [0], color=FILLED_COLORS["HE2"], linewidth=6, alpha=0.45, label="HIS HE2"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=8)
    fig.suptitle("Histidine side chain proton chemical shifts from BMRB", fontsize=12.5, y=0.965, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
    return fig


def save_figure(fig, pdf_path: Path, svg_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the proton page from cached filtered and SS compact CSVs.")
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
        default=OUTPUT_DIR / "bmrb_histogram_proton_page.pdf",
    )
    parser.add_argument("--proton-bins", type=int, default=260)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    filtered_rows = read_filtered_compact(args.filtered_csv)
    ss_rows = read_ss_compact(args.ss_csv)
    fig = make_proton_figure(filtered_rows, ss_rows, args.proton_bins)
    his_fig = make_histidine_proton_figure(filtered_rows, args.proton_bins)
    save_figure(fig, args.output_pdf, args.output_pdf.with_suffix(".svg"))
    his_pdf = args.output_pdf.with_name(f"{args.output_pdf.stem}_histidine_sidechain.pdf")
    save_figure(his_fig, his_pdf, his_pdf.with_suffix(".svg"))
    plt.close(fig)
    plt.close(his_fig)
    print(f"Wrote {args.output_pdf}")
    print(f"Wrote {args.output_pdf.with_suffix('.svg')}")
    print(f"Wrote {his_pdf}")
    print(f"Wrote {his_pdf.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
