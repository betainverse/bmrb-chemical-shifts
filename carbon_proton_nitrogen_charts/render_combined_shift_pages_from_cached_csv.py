#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import render_carbon_pages_from_cached_csv as carbon
import render_nitrogen_page_from_cached_csv as nitrogen
import render_proton_page_from_cached_csv as proton

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE_DIR = PROJECT_ROOT / "bmrb_data_cache"
CHARTS_DIR = PROJECT_ROOT / "carbon_proton_nitrogen_charts"
OUTPUT_DIR = CHARTS_DIR / "output"


plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def export_page(fig, output_dir: Path, basename: str, page_number: int, slug: str) -> None:
    stem = f"{basename}_p{page_number:02d}_{slug}"
    fig.savefig(output_dir / f"{stem}.svg", format="svg")
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf")


def make_special_cases_figure(
    filtered_rows: list[dict[str, object]],
    n_bin_count: int,
    proton_bin_count: int,
    c_bin_count: int,
    image_path: Path,
):
    his_n_values = nitrogen.build_filtered_buckets(filtered_rows, nitrogen.histidine_nitrogen_bucket)
    arg_vals: list[float] = []
    for row in filtered_rows:
        if str(row["residue_3"]).upper() != "ARG":
            continue
        atom = carbon.filtered_trace_atom(row)
        if atom.startswith("CZ"):
            arg_vals.extend(float(v) for v in row["shifts"] if 150.0 <= float(v) <= 170.0)  # type: ignore[arg-type]

    his_nd1_vals = [v for v in his_n_values.get(("HIS", "ND"), []) if 150.0 <= v <= 200.0]
    his_ne2_vals = [v for v in his_n_values.get(("HIS", "NE2_SIDE"), []) if 150.0 <= v <= 200.0]
    his_hd1_vals: list[float] = []
    his_he2_vals: list[float] = []
    for row in filtered_rows:
        if str(row["residue_3"]).upper() != "HIS":
            continue
        atom = str(row["trace_atom"]).upper()
        vals = [float(v) for v in row["shifts"] if 6.0 <= float(v) <= 16.0]  # type: ignore[arg-type]
        if atom == "HD1":
            his_hd1_vals.extend(vals)
        elif atom == "HE2":
            his_he2_vals.extend(vals)

    arg_bins = np.linspace(150.0, 170.0, c_bin_count + 1)
    his_n_step = 50.0 / max(n_bin_count, 1)
    his_n_bins = np.arange(150.0, 200.0 + his_n_step, his_n_step)
    his_h_bins = np.linspace(6.0, 16.0, proton_bin_count + 1)

    fig = plt.figure(figsize=(8.5, 11))
    row_height = nitrogen.main_page_axis_height()
    chart_width = 0.46
    left = 0.31
    ax_arg = fig.add_axes([left, 0.84, chart_width, row_height])
    ax_his_n = fig.add_axes([left, 0.74, chart_width, row_height])
    ax_his_h = fig.add_axes([left, 0.64, chart_width, row_height])
    ax_his_he2 = ax_his_h.twinx()

    arg_counts, _ = np.histogram(arg_vals, bins=arg_bins)
    arg_ymax = max(int(arg_counts.max()) if len(arg_counts) else 0, 1)
    if arg_vals:
        ax_arg.hist(
            arg_vals,
            bins=arg_bins,
            histtype="stepfilled",
            color=carbon.FILLED_COLORS["TRP_CZ"],
            edgecolor=carbon.FILLED_COLORS["TRP_CZ"],
            alpha=0.45,
            linewidth=0.8,
        )
    ax_arg.set_xlim(170, 150)
    ax_arg.set_ylim(0, arg_ymax * 1.05)
    ax_arg.tick_params(axis="y", left=False, labelleft=False)
    ax_arg.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, labelsize=8, pad=1)
    ax_arg.xaxis.set_major_locator(MultipleLocator(10.0))
    ax_arg.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax_arg.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax_arg.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax_arg.set_xlabel("Carbon Chemical Shift (ppm)", fontsize=9, color=carbon.AXIS_COLOR, labelpad=4)
    ax_arg.text(
        -0.14,
        0.5,
        "ARG CZ",
        transform=ax_arg.transAxes,
        ha="right",
        va="center",
        fontsize=8.2,
        fontweight="semibold",
        fontfamily="DejaVu Serif",
        color=carbon.LABEL_COLOR,
    )

    his_n_max = 1
    for vals, color in [
        (his_nd1_vals, nitrogen.FILLED_COLORS["ND"]),
        (his_ne2_vals, nitrogen.FILLED_COLORS["NE2_SIDE"]),
    ]:
        counts, _ = np.histogram(vals, bins=his_n_bins)
        his_n_max = max(his_n_max, int(counts.max()) if len(counts) else 0, 1)
        if vals:
            ax_his_n.hist(vals, bins=his_n_bins, histtype="stepfilled", color=color, edgecolor=color, alpha=0.45, linewidth=0.8)
    ax_his_n.set_xlim(200, 150)
    ax_his_n.set_ylim(0, his_n_max * 1.05)
    ax_his_n.tick_params(axis="y", left=False, labelleft=False)
    ax_his_n.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, labelsize=8, pad=1)
    ax_his_n.xaxis.set_major_locator(MultipleLocator(10.0))
    ax_his_n.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax_his_n.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax_his_n.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax_his_n.set_xlabel("Nitrogen Chemical Shift (ppm)", fontsize=9, color=nitrogen.AXIS_COLOR, labelpad=4)
    ax_his_n.text(
        -0.14,
        0.5,
        "HIS ND1 / NE2",
        transform=ax_his_n.transAxes,
        ha="right",
        va="center",
        fontsize=8.2,
        fontweight="semibold",
        fontfamily="DejaVu Serif",
        color=nitrogen.LABEL_COLOR,
    )

    hd1_counts, _ = np.histogram(his_hd1_vals, bins=his_h_bins)
    he2_counts_unweighted, _ = np.histogram(his_he2_vals, bins=his_h_bins)
    hd1_peak = int(hd1_counts.max()) if len(hd1_counts) else 0
    he2_peak = int(he2_counts_unweighted.max()) if len(he2_counts_unweighted) else 0
    he2_weight = 0.5 if he2_peak == 0 else min(1.0, 0.5 * hd1_peak / he2_peak) if hd1_peak else 0.5
    he2_weights = np.full(len(his_he2_vals), he2_weight) if his_he2_vals else None
    he2_counts, _ = np.histogram(his_he2_vals, bins=his_h_bins, weights=he2_weights)
    his_h_max = max(hd1_peak, 1)
    his_he2_max = max(int(he2_counts.max()) if len(he2_counts) else 0, hd1_peak, 1)
    if his_hd1_vals:
        ax_his_h.hist(
            his_hd1_vals,
            bins=his_h_bins,
            histtype="stepfilled",
            color=proton.FILLED_COLORS["HD"],
            edgecolor=proton.FILLED_COLORS["HD"],
            alpha=0.45,
            linewidth=0.8,
        )
    if his_he2_vals:
        ax_his_he2.hist(
            his_he2_vals,
            bins=his_h_bins,
            weights=he2_weights,
            histtype="stepfilled",
            color=proton.FILLED_COLORS["HE2"],
            edgecolor=proton.FILLED_COLORS["HE2"],
            alpha=0.45,
            linewidth=0.8,
        )
    ax_his_h.set_xlim(16, 6)
    ax_his_h.set_ylim(0, his_h_max * 1.05)
    ax_his_h.tick_params(axis="y", left=False, labelleft=False)
    ax_his_h.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, labelsize=8, pad=1)
    ax_his_h.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_his_h.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax_his_h.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_his_h.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax_his_h.set_xlabel("Hydrogen Chemical Shift (ppm)", fontsize=9, color=proton.AXIS_COLOR, labelpad=4)
    ax_his_he2.set_ylim(0, his_he2_max * 2.1)
    ax_his_he2.tick_params(axis="y", right=False, labelright=False)
    for side in ["right", "left", "top", "bottom"]:
        ax_his_he2.spines[side].set_visible(False)
    ax_his_h.text(
        -0.14,
        0.5,
        "HIS HD1 / HE2",
        transform=ax_his_h.transAxes,
        ha="right",
        va="center",
        fontsize=8.2,
        fontweight="semibold",
        fontfamily="DejaVu Serif",
        color=proton.LABEL_COLOR,
    )

    handles = [
        plt.Line2D([0], [0], color=carbon.FILLED_COLORS["TRP_CZ"], linewidth=6, alpha=0.45, label="ARG CZ"),
        plt.Line2D([0], [0], color=nitrogen.FILLED_COLORS["ND"], linewidth=6, alpha=0.45, label="HIS ND1"),
        plt.Line2D([0], [0], color=nitrogen.FILLED_COLORS["NE2_SIDE"], linewidth=6, alpha=0.45, label="HIS NE2"),
        plt.Line2D([0], [0], color=proton.FILLED_COLORS["HD"], linewidth=6, alpha=0.45, label="HIS HD1"),
        plt.Line2D([0], [0], color=proton.FILLED_COLORS["HE2"], linewidth=6, alpha=0.45, label="HIS HE2"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955), fontsize=8)
    fig.suptitle("Special-case chemical shifts from BMRB", fontsize=12.5, y=0.968, fontweight="semibold", fontfamily="DejaVu Serif", color=carbon.LABEL_COLOR)

    ax_img = fig.add_axes([0.09, 0.07, 0.82, 0.50])
    ax_img.axis("off")
    if image_path.exists():
        img = imread(image_path)
        ax_img.imshow(img, interpolation="none", resample=False)
        fig.text(
            0.09,
            0.04,
            "Reference image: CYANA Standard Nomenclature, https://cyana.org/wiki/Standard_CYANA_nomenclature",
            fontsize=7.2,
            color=carbon.AXIS_COLOR,
            ha="left",
            va="bottom",
        )
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the combined cached BMRB histogram document.")
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
        default=OUTPUT_DIR / "bmrb_histogram_combined_pages.pdf",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=OUTPUT_DIR / "bmrb_histogram_combined_pages_exports",
    )
    parser.add_argument("--ca-cb-bins", type=int, default=220)
    parser.add_argument("--c-bins", type=int, default=160)
    parser.add_argument("--sidechain-bins", type=int, default=220)
    parser.add_argument("--aromatic-bins", type=int, default=140)
    parser.add_argument("--proton-bins", type=int, default=260)
    parser.add_argument("--nitrogen-bins", type=int, default=160)
    parser.add_argument(
        "--cyana-image",
        type=Path,
        default=PROJECT_ROOT / "CyanaNomenclature.png",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    filtered_rows = carbon.read_filtered_compact(args.filtered_csv)
    ss_rows = carbon.read_ss_compact(args.ss_csv)

    pages = [
        ("overlay", carbon.make_overlay_figure(filtered_rows, ss_rows, args.ca_cb_bins, args.c_bins)),
        ("carbon_sidechains", carbon.make_sidechain_carbon_figure(filtered_rows, ss_rows, args.sidechain_bins, args.aromatic_bins)),
        ("protons", proton.make_proton_figure(filtered_rows, ss_rows, args.proton_bins)),
        ("nitrogen", nitrogen.make_nitrogen_figure(filtered_rows, ss_rows, args.nitrogen_bins)),
        ("special_cases", make_special_cases_figure(filtered_rows, args.nitrogen_bins, args.proton_bins, args.c_bins, args.cyana_image)),
    ]
    if not args.cyana_image.exists():
        print(f"Rendering special-cases page without the optional reference image: {args.cyana_image}")

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.exports_dir.mkdir(parents=True, exist_ok=True)
    basename = args.output_pdf.stem
    with PdfPages(args.output_pdf) as pdf:
        for idx, (slug, fig) in enumerate(pages, start=1):
            pdf.savefig(fig)
            export_page(fig, args.exports_dir, basename, idx, slug)
            plt.close(fig)

    print(f"Wrote {args.output_pdf}")
    print(f"Wrote page exports in {args.exports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
