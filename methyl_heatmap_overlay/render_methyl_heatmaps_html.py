#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean

import numpy as np
from plotly import graph_objects as go
from plotly import io as pio
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_DIR = PROJECT_ROOT / "methyl_heatmap_overlay"
DEFAULT_SHIFT_TABLE_CSV = (
    PROJECT_ROOT / "bmrb_data_cache" / "unfiltered_assignment_level_dataset" / "bmrb_shift_table_full.csv"
)


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


def read_shift_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def trim_outliers(points: list[tuple[float, float, str]], percentile: float = 0.5) -> list[tuple[float, float, str]]:
    xs = np.array([x for x, _, _ in points], dtype=float)
    ys = np.array([y for _, y, _ in points], dtype=float)
    x_low, x_high = np.percentile(xs, [percentile, 100.0 - percentile])
    y_low, y_high = np.percentile(ys, [percentile, 100.0 - percentile])
    return [(x, y, label) for x, y, label in points if x_low <= x <= x_high and y_low <= y <= y_high]


def white_to_color_scale(hex_color: str) -> list[list[object]]:
    transparent = hex_to_rgba(hex_color, 0.0)
    return [
        [0.0, transparent],
        [0.01, hex_to_rgba(hex_color, 0.03)],
        [0.04, hex_to_rgba(hex_color, 0.10)],
        [0.12, hex_to_rgba(hex_color, 0.22)],
        [0.35, hex_to_rgba(hex_color, 0.48)],
        [1.0, hex_to_rgba(hex_color, 0.92)],
    ]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_density_grid(points: list[tuple[float, float, str]], bins: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array([x for x, _, _ in points], dtype=float)
    ys = np.array([y for _, y, _ in points], dtype=float)
    x_edges = np.linspace(GLOBAL_X_RANGE[0], GLOBAL_X_RANGE[1], bins + 1)
    y_edges = np.linspace(GLOBAL_Y_RANGE[0], GLOBAL_Y_RANGE[1], bins + 1)
    hist, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    return x_centers, y_centers, hist.T


def make_overlay_figure(spec_points: list[tuple[HeatmapSpec, list[tuple[float, float, str]]]]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.84, 0.16],
        row_heights=[0.16, 0.84],
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
        specs=[
            [{"type": "histogram"}, {"type": "xy"}],
            [{"type": "heatmap"}, {"type": "histogram"}],
        ],
    )

    for spec, points in spec_points:
        xs = [x for x, _, _ in points]
        ys = [y for _, y, _ in points]
        color = TYPE_COLORS[spec.slug]
        x_centers, y_centers, z_grid = build_density_grid(points, bins=160)
        fig.add_trace(
            go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=z_grid,
                colorscale=white_to_color_scale(color),
                showscale=False,
                zmin=0,
                zmax=max(float(z_grid.max()), 1.0),
                hoverongaps=False,
                opacity=1.0,
                name=spec.title,
                legendgroup=spec.slug,
                hovertemplate=f"{spec.title}<br>x=%{{x:.3f}} ppm<br>y=%{{y:.3f}} ppm<br>count=%{{z:.0f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=xs,
                nbinsx=72,
                marker={"color": hex_to_rgba(color, 0.58)},
                opacity=0.58,
                legendgroup=spec.slug,
                showlegend=False,
                hovertemplate=f"{spec.title}<br>x=%{{x:.3f}} ppm<br>count=%{{y}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                y=ys,
                nbinsy=72,
                marker={"color": hex_to_rgba(color, 0.58)},
                opacity=0.58,
                legendgroup=spec.slug,
                showlegend=False,
                hovertemplate=f"{spec.title}<br>y=%{{y:.3f}} ppm<br>count=%{{x}}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        template="none",
        height=920,
        width=1180,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 80, "r": 36, "t": 102, "b": 62},
        title={
            "text": "Overlayed methyl proton-carbon density map",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 28, "family": "DejaVu Serif"},
        },
        hovermode="closest",
        barmode="overlay",
        bargap=0.02,
        legend={
            "orientation": "h",
            "x": 0.02,
            "y": 1.02,
            "xanchor": "left",
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.90)",
        },
    )

    fig.update_xaxes(
        range=[GLOBAL_X_RANGE[1], GLOBAL_X_RANGE[0]],
        gridcolor="#e5e7eb",
        zeroline=False,
        title_text="Methyl proton chemical shift (ppm)",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[GLOBAL_Y_RANGE[1], GLOBAL_Y_RANGE[0]],
        gridcolor="#e5e7eb",
        zeroline=False,
        title_text="Methyl carbon chemical shift (ppm)",
        row=2,
        col=1,
    )
    fig.update_xaxes(range=[GLOBAL_X_RANGE[1], GLOBAL_X_RANGE[0]], showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(range=[GLOBAL_Y_RANGE[1], GLOBAL_Y_RANGE[0]], showticklabels=False, row=2, col=2)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    return fig


def write_html(
    spec_points: list[tuple[HeatmapSpec, list[tuple[float, float, str]]]],
    output_path: Path,
) -> None:
    summary_items = "\n".join(
        f"<li><strong>{escape(spec.title)}</strong>: {len(points):,} paired residue instances</li>" for spec, points in spec_points
    )
    fig = make_overlay_figure(spec_points)
    fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BMRB methyl heatmaps</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --line: #dbe2ea;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "DejaVu Sans", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      width: min(1260px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    .hero {{
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: "DejaVu Serif", serif;
      font-size: clamp(2rem, 3vw, 2.8rem);
      line-height: 1.05;
    }}
    .lede {{
      margin: 0;
      max-width: 82ch;
      color: var(--muted);
      line-height: 1.5;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px 20px;
      margin-top: 18px;
    }}
    .chart-panel {{
      padding: 10px 10px 0;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.7;
    }}
    .legend-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px 14px;
      margin-top: 16px;
    }}
    .legend-chip {{
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, 0.10);
      flex: 0 0 auto;
    }}
    .plotly-graph-div {{
      width: 100%;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>BMRB Methyl Density Overlay</h1>
      <p class="lede">All six requested methyl classes are layered onto one shared proton-carbon map so the occupied regions can be compared directly instead of looking at isolated charts in unrelated frames.</p>
    </section>
    <section class="panel">
      <ul>
        {summary_items}
      </ul>
      <div class="legend-grid">
        {''.join(f'<div class="legend-chip"><span class="swatch" style="background:{TYPE_COLORS[spec.slug]};"></span>{escape(spec.title)}</div>' for spec, _ in spec_points)}
      </div>
    </section>
    <section class="panel chart-panel">
      {fig_html}
    </section>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render methyl proton/carbon overlay heatmaps as standalone HTML.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_SHIFT_TABLE_CSV)
    parser.add_argument(
        "--output-html",
        type=Path,
        default=OVERLAY_DIR / "bmrb_methyl_heatmaps.html",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_shift_rows(args.csv)
    grouped = build_residue_atom_values(rows)
    spec_points = [(spec, trim_outliers(extract_points(grouped, spec))) for spec in SPECS]
    missing = [spec.title for spec, points in spec_points if not points]
    if missing:
        raise RuntimeError(f"No paired points found for: {', '.join(missing)}")
    write_html(spec_points, args.output_html)
    print(f"Wrote {args.output_html}")
    for spec, points in spec_points:
        print(f"{spec.slug}: {len(points)} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
