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

NON_AROMATIC_CARBON_RESIDUES = {
    "CG": {"ARG", "GLN", "GLU", "ILE", "LEU", "LYS", "MET", "PRO", "THR", "VAL"},
    "CD": {"ILE", "LEU", "LYS", "PRO"},
    "CE": {"LYS", "MET"},
}

AROMATIC_CARBON_RESIDUES = {
    "CG": {"HIS", "PHE", "TRP", "TYR"},
    "CD": {"HIS", "PHE", "TRP", "TYR"},
    "CE": {"HIS", "PHE", "TRP", "TYR"},
}

SIDECHAIN_CARBONYL_ATOMS = {
    ("ASN", "CG"),
    ("ASP", "CG"),
    ("GLN", "CD"),
    ("GLU", "CD"),
}

FILLED_COLORS = {
    "CA": "#60a5fa",
    "CB": "#f472b6",
    "CG": "#34d399",
    "CD": "#a78bfa",
    "CE": "#f59e0b",
    "C": "#9ca3af",
    "TRP_CZ": "#9ca3af",
    "TRP_CH": "#f97316",
}

SS_STYLES = {
    "C": {"helix_fill": "#374151", "sheet_fill": "#9ca3af", "coil_outline": "#111827"},
    "CA": {"helix_fill": "#1d4ed8", "sheet_fill": "#60a5fa", "coil_outline": "#2563eb"},
    "CB": {"helix_fill": "#be185d", "sheet_fill": "#f472b6", "coil_outline": "#dc2626"},
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


def draw_ss_hist(ax, values, residue: str, atom: str, bins: np.ndarray) -> int:
    helix_vals = values.get((residue, atom, "helix"), [])
    sheet_vals = values.get((residue, atom, "sheet"), [])
    coil_vals = values.get((residue, atom, "coil"), [])
    styles = SS_STYLES[atom]
    counts = []
    for vals in [helix_vals, sheet_vals, coil_vals]:
        hist_counts, _ = np.histogram(vals, bins=bins)
        counts.append(int(hist_counts.max()) if len(hist_counts) else 0)
    ymax = max(max(counts), 1)
    if helix_vals:
        ax.hist(helix_vals, bins=bins, histtype="stepfilled", color=styles["helix_fill"], edgecolor=styles["helix_fill"], alpha=0.36, linewidth=0.8)
    if sheet_vals:
        ax.hist(sheet_vals, bins=bins, histtype="stepfilled", color=styles["sheet_fill"], edgecolor=styles["sheet_fill"], alpha=0.36, linewidth=0.8)
    if coil_vals:
        ax.hist(coil_vals, bins=bins, histtype="step", color=styles["coil_outline"], linewidth=0.9, linestyle=":")
    return ymax


def main_page_axis_height() -> float:
    return 0.78 / (len(AA_ORDER) + 0.18 * (len(AA_ORDER) - 1))


def aromatic_axis_width() -> float:
    total_width = 0.90 - 0.14
    average_width = total_width / (2.0 + 0.20)
    spacing = 0.20 * average_width
    return (total_width - spacing) / 3.0


def build_ss_values(ss_rows: list[dict[str, object]], atoms: set[str]) -> dict[tuple[str, str, str], list[float]]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in ss_rows:
        residue = str(row["residue_3"])
        atom = str(row["page_atom"]).upper()
        ss = str(row["secondary_structure"])
        if atom not in atoms or ss not in {"helix", "sheet", "coil"}:
            continue
        values[(residue, atom, ss)].extend(row["shifts"])  # type: ignore[arg-type]
    return values


def filtered_trace_atom(row: dict[str, object]) -> str:
    residue = str(row["residue_3"]).upper()
    page_atom = str(row["page_atom"]).upper()
    trace_atom = str(row["trace_atom"]).upper()
    trace_name = str(row["trace_name"]).upper()
    if trace_name.startswith(f"{residue}-"):
        trace_atom = trace_name.split("-", 1)[1]
    return trace_atom if trace_atom else page_atom


def non_aromatic_bucket(atom: str, residue: str) -> str | None:
    atom = atom.upper()
    residue = residue.upper()
    if atom == "CA":
        return "CA"
    if atom == "CB":
        return "CB"
    if atom.startswith("CG"):
        if residue not in NON_AROMATIC_CARBON_RESIDUES["CG"]:
            return None
        return "CG"
    if atom.startswith("CD"):
        if residue not in NON_AROMATIC_CARBON_RESIDUES["CD"]:
            return None
        return "CD"
    if atom.startswith("CE"):
        if residue not in NON_AROMATIC_CARBON_RESIDUES["CE"]:
            return None
        return "CE"
    return None


def aromatic_bucket(atom: str, residue: str) -> str | None:
    atom = atom.upper()
    residue = residue.upper()
    if residue not in {"PHE", "TYR", "TRP", "HIS"}:
        return None
    if residue == "TRP" and atom.startswith("CH"):
        return "TRP_CH"
    if residue == "TRP" and atom.startswith("CZ"):
        return "TRP_CZ"
    if atom.startswith("CG"):
        return "CG" if residue in AROMATIC_CARBON_RESIDUES["CG"] else None
    if atom.startswith("CD"):
        return "CD" if residue in AROMATIC_CARBON_RESIDUES["CD"] else None
    if atom.startswith("CE"):
        return "CE" if residue in AROMATIC_CARBON_RESIDUES["CE"] else None
    return None


def build_filtered_bucketed(filtered_rows: list[dict[str, object]], bucket_func) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in filtered_rows:
        residue = str(row["residue_3"])
        atom = filtered_trace_atom(row)
        bucket = bucket_func(atom, residue)
        if bucket is None:
            continue
        values[(residue, bucket)].extend(row["shifts"])  # type: ignore[arg-type]
    return values


def build_carbonyl_values(filtered_rows: list[dict[str, object]], ss_rows: list[dict[str, object]]) -> dict[tuple[str, str, str], list[float]]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in ss_rows:
        residue = str(row["residue_3"])
        atom = str(row["page_atom"]).upper()
        ss = str(row["secondary_structure"])
        if atom != "C" or ss not in {"helix", "sheet", "coil"}:
            continue
        vals = [float(v) for v in row["shifts"] if 165.0 <= float(v) <= 185.0]  # type: ignore[arg-type]
        values[(residue, "C", ss)].extend(vals)
    for row in filtered_rows:
        residue = str(row["residue_3"])
        page_atom = str(row["page_atom"]).upper()
        if (residue, page_atom) not in SIDECHAIN_CARBONYL_ATOMS:
            continue
        vals = [float(v) for v in row["shifts"] if 165.0 <= float(v) <= 185.0]  # type: ignore[arg-type]
        if not vals:
            continue
        bucket = "SC_CG" if (residue, page_atom) in {("ASN", "CG"), ("ASP", "CG")} else "SC_CD"
        values[(residue, bucket, "all")].extend(vals)
    return values


def make_overlay_figure(filtered_rows: list[dict[str, object]], ss_rows: list[dict[str, object]], cacb_bin_count: int, c_bin_count: int):
    carbonyl_values = build_carbonyl_values(filtered_rows, ss_rows)
    cacb_values = build_ss_values(ss_rows, {"CA", "CB"})
    fig, axes = plt.subplots(nrows=len(AA_ORDER), ncols=2, figsize=(8.5, 11), sharex="col", gridspec_kw={"width_ratios": [1, 2]})
    fig.subplots_adjust(left=0.14, right=0.90, top=0.84, bottom=0.06, wspace=0.20, hspace=0.18)
    c_bins = np.linspace(165, 185, c_bin_count + 1)
    cacb_bins = np.linspace(10, 75, cacb_bin_count + 1)
    for row_idx, (residue, one) in enumerate(AA_ORDER):
        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]
        ax_sc_cg = ax_left.twinx()
        ax_sc_cd = ax_left.twinx()
        ax_ca = ax_right.twinx()
        ax_cb = ax_right.twinx()
        left_max = draw_ss_hist(ax_left, carbonyl_values, residue, "C", c_bins)
        sc_cg = carbonyl_values.get((residue, "SC_CG", "all"), [])
        sc_cd = carbonyl_values.get((residue, "SC_CD", "all"), [])
        ca_max = draw_ss_hist(ax_ca, cacb_values, residue, "CA", cacb_bins)
        cb_max = draw_ss_hist(ax_cb, cacb_values, residue, "CB", cacb_bins)
        sc_cg_counts, _ = np.histogram(sc_cg, bins=c_bins)
        sc_cd_counts, _ = np.histogram(sc_cd, bins=c_bins)
        sc_cg_max = max(int(sc_cg_counts.max()) if len(sc_cg_counts) else 0, 1)
        sc_cd_max = max(int(sc_cd_counts.max()) if len(sc_cd_counts) else 0, 1)
        if sc_cg:
            ax_sc_cg.hist(sc_cg, bins=c_bins, histtype="stepfilled", color=FILLED_COLORS["CG"], edgecolor=FILLED_COLORS["CG"], alpha=0.40, linewidth=0.8)
        if sc_cd:
            ax_sc_cd.hist(sc_cd, bins=c_bins, histtype="stepfilled", color=FILLED_COLORS["CD"], edgecolor=FILLED_COLORS["CD"], alpha=0.40, linewidth=0.8)
        ax_left.set_xlim(185, 165)
        ax_right.set_xlim(75, 10)
        ax_left.set_ylim(0, left_max * 1.05)
        ax_right.set_ylim(0, 1)
        ax_left.tick_params(axis="y", left=False, labelleft=False)
        ax_right.tick_params(axis="y", left=False, labelleft=False)
        for twin, ymax in [(ax_sc_cg, sc_cg_max), (ax_sc_cd, sc_cd_max), (ax_ca, ca_max), (ax_cb, cb_max)]:
            twin.set_ylim(0, ymax * 1.05)
            twin.tick_params(axis="y", right=False, labelright=False)
            for side in ["right", "left", "top", "bottom"]:
                twin.spines[side].set_visible(False)
        ax_left.text(-0.14, 0.5, f"{residue} ({one})", transform=ax_left.transAxes, ha="right", va="center", fontsize=7.8, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
        if residue == "CYS":
            ax_cb.annotate("reduced", xy=(30.0, cb_max * 0.72), xytext=(33.2, cb_max * 0.83), fontsize=6.5, color="#000000", arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8})
            ax_cb.annotate("oxidized", xy=(40.0, cb_max * 0.45), xytext=(43.3, cb_max * 0.64), fontsize=6.5, color="#000000", arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8})
        if row_idx != len(AA_ORDER) - 1:
            ax_left.tick_params(axis="x", labelbottom=False)
            ax_right.tick_params(axis="x", labelbottom=False)
        else:
            ax_left.tick_params(axis="x", labelsize=7)
            ax_right.tick_params(axis="x", labelsize=7)
    axes[0, 0].set_title("Carbonyl / Carboxyl / Carbamoyl", fontsize=10.5, pad=6, fontweight="semibold")
    axes[0, 1].set_title("CA and CB", fontsize=10.5, pad=6, fontweight="semibold")
    axes[0, 0].tick_params(axis="x", top=True, labeltop=True, labelsize=7, pad=1)
    axes[0, 1].tick_params(axis="x", top=True, labeltop=True, labelsize=7, pad=1)
    for row_axes in axes:
        row_axes[0].xaxis.set_major_locator(MultipleLocator(5.0))
        row_axes[0].xaxis.set_major_formatter(FormatStrFormatter("%d"))
        row_axes[0].xaxis.set_minor_locator(MultipleLocator(1.0))
        row_axes[0].tick_params(axis="x", which="minor", length=2, width=0.5)
        row_axes[1].xaxis.set_major_locator(MultipleLocator(10.0))
        row_axes[1].xaxis.set_major_formatter(FormatStrFormatter("%d"))
        row_axes[1].xaxis.set_minor_locator(MultipleLocator(1.0))
        row_axes[1].tick_params(axis="x", which="minor", length=2, width=0.5)
    handles = [
        plt.Line2D([0], [0], color=SS_STYLES["C"]["helix_fill"], linewidth=6, alpha=0.6, label="C alpha"),
        plt.Line2D([0], [0], color=SS_STYLES["C"]["sheet_fill"], linewidth=6, alpha=0.6, label="C beta"),
        plt.Line2D([0], [0], color=SS_STYLES["C"]["coil_outline"], linewidth=0.9, linestyle=":", label="C coil"),
        plt.Line2D([0], [0], color=FILLED_COLORS["CG"], linewidth=6, alpha=0.45, label="ASN/ASP CG"),
        plt.Line2D([0], [0], color=FILLED_COLORS["CD"], linewidth=6, alpha=0.45, label="GLN/GLU CD"),
        plt.Line2D([0], [0], color=SS_STYLES["CA"]["helix_fill"], linewidth=6, alpha=0.6, label="CA alpha"),
        plt.Line2D([0], [0], color=SS_STYLES["CA"]["sheet_fill"], linewidth=6, alpha=0.6, label="CA beta"),
        plt.Line2D([0], [0], color=SS_STYLES["CA"]["coil_outline"], linewidth=0.9, linestyle=":", label="CA coil"),
        plt.Line2D([0], [0], color=SS_STYLES["CB"]["helix_fill"], linewidth=6, alpha=0.6, label="CB alpha"),
        plt.Line2D([0], [0], color=SS_STYLES["CB"]["sheet_fill"], linewidth=6, alpha=0.6, label="CB beta"),
        plt.Line2D([0], [0], color=SS_STYLES["CB"]["coil_outline"], linewidth=0.9, linestyle=":", label="CB coil"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.948), fontsize=8)
    fig.suptitle("Backbone and side-chain carbon chemical shifts from BMRB", fontsize=12.5, y=0.975, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
    fig.supxlabel("Carbon Chemical Shift (ppm)", fontsize=10, color=AXIS_COLOR)
    return fig


def make_sidechain_carbon_figure(filtered_rows: list[dict[str, object]], ss_rows: list[dict[str, object]], sidechain_bin_count: int, aromatic_bin_count: int):
    aromatic_values = build_filtered_bucketed(filtered_rows, aromatic_bucket)
    non_aromatic_values = build_filtered_bucketed(filtered_rows, non_aromatic_bucket)
    ss_cacb = build_ss_values(ss_rows, {"CA", "CB"})
    fig, axes = plt.subplots(nrows=len(AA_ORDER), ncols=2, figsize=(8.5, 11), sharex="col", gridspec_kw={"width_ratios": [1, 2]})
    fig.subplots_adjust(left=0.14, right=0.90, top=0.84, bottom=0.06, wspace=0.20, hspace=0.18)
    aromatic_bins = np.linspace(105.0, 145.0, aromatic_bin_count + 1)
    aliphatic_bins = np.linspace(10.0, 75.0, sidechain_bin_count + 1)
    for row_idx, (residue, one) in enumerate(AA_ORDER):
        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]
        left_axes = {key: ax_left.twinx() for key in ["CG", "CD", "CE", "TRP_CZ", "TRP_CH"]}
        right_axes = {key: ax_right.twinx() for key in ["CA", "CB", "CG", "CD", "CE"]}
        left_maxima = {}
        for key in left_axes:
            vals = aromatic_values.get((residue, key), [])
            counts, _ = np.histogram(vals, bins=aromatic_bins)
            left_maxima[key] = max(int(counts.max()) if len(counts) else 0, 1)
            if vals:
                left_axes[key].hist(vals, bins=aromatic_bins, histtype="stepfilled", color=FILLED_COLORS[key], edgecolor=FILLED_COLORS[key], alpha=0.40, linewidth=0.8)
        right_maxima = {}
        right_maxima["CA"] = draw_ss_hist(right_axes["CA"], ss_cacb, residue, "CA", aliphatic_bins)
        right_maxima["CB"] = draw_ss_hist(right_axes["CB"], ss_cacb, residue, "CB", aliphatic_bins)
        for key in ["CG", "CD", "CE"]:
            vals = non_aromatic_values.get((residue, key), [])
            counts, _ = np.histogram(vals, bins=aliphatic_bins)
            right_maxima[key] = max(int(counts.max()) if len(counts) else 0, 1)
            if vals:
                right_axes[key].hist(vals, bins=aliphatic_bins, histtype="stepfilled", color=FILLED_COLORS[key], edgecolor=FILLED_COLORS[key], alpha=0.38, linewidth=0.8)
        ax_left.set_xlim(145, 105)
        ax_right.set_xlim(75, 10)
        ax_left.set_ylim(0, 1)
        ax_right.set_ylim(0, 1)
        ax_left.tick_params(axis="y", left=False, labelleft=False)
        ax_right.tick_params(axis="y", left=False, labelleft=False)
        for key, twin in left_axes.items():
            twin.set_ylim(0, left_maxima[key] * 1.05)
            twin.tick_params(axis="y", right=False, labelright=False)
            for side in ["right", "left", "top", "bottom"]:
                twin.spines[side].set_visible(False)
        for key, twin in right_axes.items():
            twin.set_ylim(0, right_maxima[key] * 1.05)
            twin.tick_params(axis="y", right=False, labelright=False)
            for side in ["right", "left", "top", "bottom"]:
                twin.spines[side].set_visible(False)
        ax_left.text(-0.14, 0.5, f"{residue} ({one})", transform=ax_left.transAxes, ha="right", va="center", fontsize=7.8, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
        if residue == "CYS":
            right_axes["CB"].annotate("reduced", xy=(30.0, right_maxima["CB"] * 0.72), xytext=(33.2, right_maxima["CB"] * 0.83), fontsize=6.5, color="#000000", arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8})
            right_axes["CB"].annotate("oxidized", xy=(40.0, right_maxima["CB"] * 0.45), xytext=(43.3, right_maxima["CB"] * 0.64), fontsize=6.5, color="#000000", arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8})
        if residue == "ILE":
            right_axes["CG"].annotate("CG1", xy=(27.5, right_maxima["CG"] * 0.78), xytext=(28.8, right_maxima["CG"] * 0.80), fontsize=6.5, color="#000000", arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8})
            right_axes["CG"].annotate("CG2", xy=(17.5, right_maxima["CG"] * 0.62), xytext=(18.5, right_maxima["CG"] * 0.66), fontsize=6.5, color="#000000", arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8})
        if residue == "TRP":
            left_axes["TRP_CZ"].annotate("CZ2", xy=(114.5, left_maxima["TRP_CZ"] * 0.72), xytext=(112.4, left_maxima["TRP_CZ"] * 0.60), fontsize=6.5, color="#000000", arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8})
            left_axes["TRP_CZ"].annotate("CZ3", xy=(122.0, left_maxima["TRP_CZ"] * 0.58), xytext=(122.0, left_maxima["TRP_CZ"] * 0.76), fontsize=6.5, color="#000000", ha="center", arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.8})
        if row_idx != len(AA_ORDER) - 1:
            ax_left.tick_params(axis="x", labelbottom=False)
            ax_right.tick_params(axis="x", labelbottom=False)
        else:
            ax_left.tick_params(axis="x", labelsize=7)
            ax_right.tick_params(axis="x", labelsize=7)
    axes[0, 0].set_title("Aromatic carbons", fontsize=10.5, pad=6, fontweight="semibold")
    axes[0, 1].set_title("Aliphatic carbons", fontsize=10.5, pad=6, fontweight="semibold")
    axes[0, 0].tick_params(axis="x", top=True, labeltop=True, labelsize=7, pad=1)
    axes[0, 1].tick_params(axis="x", top=True, labeltop=True, labelsize=7, pad=1)
    for row_axes in axes:
        row_axes[0].xaxis.set_major_locator(MultipleLocator(10.0))
        row_axes[0].xaxis.set_major_formatter(FormatStrFormatter("%d"))
        row_axes[0].xaxis.set_minor_locator(MultipleLocator(1.0))
        row_axes[0].tick_params(axis="x", which="minor", length=2, width=0.5)
        row_axes[1].xaxis.set_major_locator(MultipleLocator(10.0))
        row_axes[1].xaxis.set_major_formatter(FormatStrFormatter("%d"))
        row_axes[1].xaxis.set_minor_locator(MultipleLocator(1.0))
        row_axes[1].tick_params(axis="x", which="minor", length=2, width=0.5)
    handles = [
        plt.Line2D([0], [0], color=FILLED_COLORS["CG"], linewidth=6, alpha=0.40, label="Aromatic CG"),
        plt.Line2D([0], [0], color=FILLED_COLORS["CD"], linewidth=6, alpha=0.40, label="Aromatic CD"),
        plt.Line2D([0], [0], color=FILLED_COLORS["CE"], linewidth=6, alpha=0.40, label="Aromatic CE"),
        plt.Line2D([0], [0], color=FILLED_COLORS["TRP_CZ"], linewidth=6, alpha=0.40, label="ARG/TRP CZ"),
        plt.Line2D([0], [0], color=FILLED_COLORS["TRP_CH"], linewidth=6, alpha=0.40, label="TRP CH"),
        plt.Line2D([0], [0], color=SS_STYLES["CA"]["helix_fill"], linewidth=6, alpha=0.6, label="CA alpha"),
        plt.Line2D([0], [0], color=SS_STYLES["CA"]["sheet_fill"], linewidth=6, alpha=0.6, label="CA beta"),
        plt.Line2D([0], [0], color=SS_STYLES["CA"]["coil_outline"], linewidth=0.9, linestyle=":", label="CA coil"),
        plt.Line2D([0], [0], color=SS_STYLES["CB"]["helix_fill"], linewidth=6, alpha=0.6, label="CB alpha"),
        plt.Line2D([0], [0], color=SS_STYLES["CB"]["sheet_fill"], linewidth=6, alpha=0.6, label="CB beta"),
        plt.Line2D([0], [0], color=SS_STYLES["CB"]["coil_outline"], linewidth=0.9, linestyle=":", label="CB coil"),
        plt.Line2D([0], [0], color=FILLED_COLORS["CG"], linewidth=6, alpha=0.38, label="CG"),
        plt.Line2D([0], [0], color=FILLED_COLORS["CD"], linewidth=6, alpha=0.38, label="CD"),
        plt.Line2D([0], [0], color=FILLED_COLORS["CE"], linewidth=6, alpha=0.38, label="CE"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.948), fontsize=8)
    fig.suptitle("Backbone and side-chain carbon chemical shifts from BMRB", fontsize=12.5, y=0.975, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
    fig.supxlabel("Carbon Chemical Shift (ppm)", fontsize=10, color=AXIS_COLOR)
    return fig


def make_sidechain_carbon_with_arg_cz_inset_figure(
    filtered_rows: list[dict[str, object]], ss_rows: list[dict[str, object]], sidechain_bin_count: int, aromatic_bin_count: int, c_bin_count: int
):
    fig = make_sidechain_carbon_figure(filtered_rows, ss_rows, sidechain_bin_count, aromatic_bin_count)

    arg_vals: list[float] = []
    for row in filtered_rows:
        if str(row["residue_3"]).upper() != "ARG":
            continue
        atom = filtered_trace_atom(row)
        if not atom.startswith("CZ"):
            continue
        arg_vals.extend(float(v) for v in row["shifts"] if 150.0 <= float(v) <= 170.0)  # type: ignore[arg-type]

    bins = np.linspace(150.0, 170.0, c_bin_count + 1)
    counts, _ = np.histogram(arg_vals, bins=bins)
    ymax = max(int(counts.max()) if len(counts) else 0, 1)

    row_height = main_page_axis_height()
    aromatic_width = aromatic_axis_width()
    # Place the inset on the left side, aligned with the aromatic-carbon column.
    inset_left = 0.14 + aromatic_width * 0.35
    inset_bottom = 0.06 + (len(AA_ORDER) - 2) * (row_height * 1.18)
    inset_width = aromatic_width * 0.46
    inset_height = row_height * 0.92
    inset_ax = fig.add_axes(
        [inset_left, inset_bottom, inset_width, inset_height],
        facecolor="white",
        zorder=10,
    )
    inset_ax.patch.set_edgecolor("#d1d5db")
    inset_ax.patch.set_linewidth(0.8)
    inset_ax.patch.set_alpha(1.0)

    if arg_vals:
        inset_ax.hist(
            arg_vals,
            bins=bins,
            histtype="stepfilled",
            color=FILLED_COLORS["TRP_CZ"],
            edgecolor=FILLED_COLORS["TRP_CZ"],
            alpha=0.45,
            linewidth=0.8,
        )

    inset_ax.set_xlim(170, 150)
    inset_ax.set_ylim(0, ymax * 1.05)
    inset_ax.tick_params(axis="y", left=False, labelleft=False)
    inset_ax.xaxis.set_major_locator(MultipleLocator(10.0))
    inset_ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    inset_ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    inset_ax.tick_params(axis="x", which="major", labelsize=5.5, pad=1, length=2.5)
    inset_ax.tick_params(axis="x", which="minor", length=1.3, width=0.45)
    inset_ax.text(0.5, 0.92, "ARG CZ", transform=inset_ax.transAxes, ha="center", va="top", fontsize=5.8, fontweight="semibold", color=LABEL_COLOR)

    return fig


def make_arg_cz_figure(filtered_rows: list[dict[str, object]], c_bin_count: int):
    vals: list[float] = []
    for row in filtered_rows:
        if str(row["residue_3"]).upper() != "ARG":
            continue
        atom = filtered_trace_atom(row)
        if not atom.startswith("CZ"):
            continue
        vals.extend(float(v) for v in row["shifts"] if 150.0 <= float(v) <= 170.0)  # type: ignore[arg-type]

    bins = np.linspace(150.0, 170.0, c_bin_count + 1)
    fig = plt.figure(figsize=(8.5, 11))
    width = aromatic_axis_width()
    height = main_page_axis_height()
    ax = fig.add_axes([0.27, 0.82, width, height])

    counts, _ = np.histogram(vals, bins=bins)
    ymax = max(int(counts.max()) if len(counts) else 0, 1)
    if vals:
        ax.hist(vals, bins=bins, histtype="stepfilled", color=FILLED_COLORS["TRP_CZ"], edgecolor=FILLED_COLORS["TRP_CZ"], alpha=0.45, linewidth=0.8)

    ax.set_xlim(170, 150)
    ax.set_ylim(0, ymax * 1.05)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", top=True, labeltop=True, labelsize=7, pad=1)
    ax.xaxis.set_major_locator(MultipleLocator(10.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax.set_xlabel("Carbon Chemical Shift (ppm)", fontsize=9, color=AXIS_COLOR, labelpad=4)
    ax.text(-0.16, 0.5, "ARG CZ", transform=ax.transAxes, ha="right", va="center", fontsize=8.2, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)

    handles = [plt.Line2D([0], [0], color=FILLED_COLORS["TRP_CZ"], linewidth=6, alpha=0.45, label="ARG/TRP CZ")]
    fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=8)
    fig.suptitle("Arginine CZ chemical shifts from BMRB", fontsize=12.5, y=0.965, fontweight="semibold", fontfamily="DejaVu Serif", color=LABEL_COLOR)
    return fig


def save_figure(fig, pdf_path: Path, svg_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the carbon pages from cached filtered and SS compact CSVs.")
    parser.add_argument("--filtered-csv", type=Path, default=DATA_CACHE_DIR / "filtered_histogram_cache_compact.csv")
    parser.add_argument("--ss-csv", type=Path, default=DATA_CACHE_DIR / "ss_histogram_cache_compact.csv")
    parser.add_argument("--base-output", type=Path, default=OUTPUT_DIR / "bmrb_histogram_carbon_page.pdf")
    parser.add_argument("--ca-cb-bins", type=int, default=220)
    parser.add_argument("--c-bins", type=int, default=160)
    parser.add_argument("--sidechain-bins", type=int, default=220)
    parser.add_argument("--aromatic-bins", type=int, default=140)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    filtered_rows = read_filtered_compact(args.filtered_csv)
    ss_rows = read_ss_compact(args.ss_csv)
    overlay_fig = make_overlay_figure(filtered_rows, ss_rows, args.ca_cb_bins, args.c_bins)
    sidechain_fig = make_sidechain_carbon_figure(filtered_rows, ss_rows, args.sidechain_bins, args.aromatic_bins)
    sidechain_inset_fig = make_sidechain_carbon_with_arg_cz_inset_figure(
        filtered_rows, ss_rows, args.sidechain_bins, args.aromatic_bins, args.c_bins
    )
    arg_fig = make_arg_cz_figure(filtered_rows, args.c_bins)
    overlay_pdf = args.base_output
    overlay_svg = overlay_pdf.with_suffix(".svg")
    sidechain_pdf = args.base_output.with_name(f"{args.base_output.stem}_sidechains.pdf")
    sidechain_svg = sidechain_pdf.with_suffix(".svg")
    sidechain_inset_pdf = args.base_output.with_name(f"{args.base_output.stem}_sidechains_arg_cz_inset.pdf")
    sidechain_inset_svg = sidechain_inset_pdf.with_suffix(".svg")
    arg_pdf = args.base_output.with_name(f"{args.base_output.stem}_arg_cz.pdf")
    arg_svg = arg_pdf.with_suffix(".svg")
    save_figure(overlay_fig, overlay_pdf, overlay_svg)
    save_figure(sidechain_fig, sidechain_pdf, sidechain_svg)
    save_figure(sidechain_inset_fig, sidechain_inset_pdf, sidechain_inset_svg)
    save_figure(arg_fig, arg_pdf, arg_svg)
    plt.close(overlay_fig)
    plt.close(sidechain_fig)
    plt.close(sidechain_inset_fig)
    plt.close(arg_fig)
    print(f"Wrote {overlay_pdf}")
    print(f"Wrote {overlay_svg}")
    print(f"Wrote {sidechain_pdf}")
    print(f"Wrote {sidechain_svg}")
    print(f"Wrote {sidechain_inset_pdf}")
    print(f"Wrote {sidechain_inset_svg}")
    print(f"Wrote {arg_pdf}")
    print(f"Wrote {arg_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
