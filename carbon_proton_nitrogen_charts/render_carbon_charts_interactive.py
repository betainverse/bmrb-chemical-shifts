#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
from pathlib import Path

from plotly import graph_objects as go
from plotly import io as pio
from plotly.subplots import make_subplots

from render_carbon_pages_from_cached_csv import AA_ORDER, filtered_trace_atom, read_filtered_compact, read_ss_compact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE_DIR = PROJECT_ROOT / "bmrb_data_cache"
CHARTS_DIR = PROJECT_ROOT / "carbon_proton_nitrogen_charts"
OUTPUT_DIR = CHARTS_DIR / "output"


PLOTLY_STATE_COLORS = {
    "helix": "rgba(29, 78, 216, 0.68)",
    "sheet": "rgba(244, 114, 182, 0.58)",
    "coil": "rgba(17, 24, 39, 0.85)",
    "all": "rgba(75, 85, 99, 0.68)",
}


def atom_sort_key(atom: str) -> tuple[int, int, str]:
    atom = atom.upper()
    prefix_order = {
        "C": 0,
        "CA": 1,
        "CB": 2,
        "CG": 3,
        "CD": 4,
        "CE": 5,
        "CZ": 6,
        "CH": 7,
    }
    prefix = atom
    suffix = ""
    while prefix and prefix[-1].isdigit():
        suffix = prefix[-1] + suffix
        prefix = prefix[:-1]
    suffix_num = int(suffix) if suffix else 0
    return (prefix_order.get(prefix, 99), suffix_num, atom)


def build_interactive_values(
    filtered_rows: list[dict[str, object]], ss_rows: list[dict[str, object]]
) -> dict[str, dict[str, dict[str, list[float]]]]:
    values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in ss_rows:
        residue = str(row["residue_3"]).upper()
        atom = str(row["page_atom"]).upper()
        if not atom.startswith("C"):
            continue
        state = str(row["secondary_structure"]).lower()
        if state not in {"helix", "sheet", "coil"}:
            continue
        values[residue][atom][state].extend(float(v) for v in row["shifts"])  # type: ignore[arg-type]
    for row in filtered_rows:
        residue = str(row["residue_3"]).upper()
        atom = filtered_trace_atom(row)
        if not atom.startswith("C"):
            continue
        if values[residue][atom]:
            continue
        values[residue][atom]["all"].extend(float(v) for v in row["shifts"])  # type: ignore[arg-type]
    return values


def make_residue_figure(
    residue: str,
    one_letter: str,
    residue_values: dict[str, dict[str, list[float]]],
    bin_size: float,
) -> go.Figure:
    atoms = sorted(
        [atom for atom, state_map in residue_values.items() if any(state_map.values())],
        key=atom_sort_key,
    )
    fig = make_subplots(rows=len(atoms), cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=atoms)
    legend_seen: set[str] = set()
    for row_idx, atom in enumerate(atoms, start=1):
        state_maps = residue_values[atom]
        state_order = [state for state in ["helix", "sheet", "coil", "all"] if state_maps.get(state)]
        for state in state_order:
            fig.add_trace(
                go.Histogram(
                    x=state_maps[state],
                    xbins={"start": 0, "end": 185, "size": bin_size},
                    name=state.title() if state != "all" else "All assignments",
                    legendgroup=state,
                    showlegend=state not in legend_seen,
                    marker={"color": PLOTLY_STATE_COLORS[state]},
                    opacity=0.72 if state != "coil" else 1.0,
                    hovertemplate=f"{atom} {state}<br>%{{x:.2f}} ppm<br>count=%{{y}}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )
            legend_seen.add(state)
        fig.update_yaxes(title_text=atom, row=row_idx, col=1, showgrid=False, zeroline=False)

    fig.update_layout(
        template="none",
        barmode="overlay",
        bargap=0.02,
        height=max(240, 118 * len(atoms) + 70),
        width=1400,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 90, "r": 24, "t": 72, "b": 48},
        title={
            "text": f"{residue} ({one_letter}) carbon chemical shifts",
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 24, "family": "DejaVu Serif"},
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.01},
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="Carbon chemical shift (ppm)",
        autorange="reversed",
        showline=True,
        linewidth=1.0,
        linecolor="#9ca3af",
        gridcolor="#e5e7eb",
        zeroline=False,
        ticks="outside",
        tickcolor="#9ca3af",
        minor={"showgrid": False},
    )
    for annotation in fig.layout.annotations:
        annotation.font.size = 13
        annotation.font.family = "DejaVu Sans"
        annotation.x = 0.0
        annotation.xanchor = "left"
    return fig


def write_interactive_html(
    filtered_rows: list[dict[str, object]],
    ss_rows: list[dict[str, object]],
    output_path: Path,
    bin_size: float,
) -> None:
    residue_values = build_interactive_values(filtered_rows, ss_rows)
    sections: list[str] = []
    for residue, one_letter in AA_ORDER:
        if residue not in residue_values:
            continue
        fig = make_residue_figure(residue, one_letter, residue_values[residue], bin_size)
        html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})
        sections.append(
            f"""
            <section class="residue-panel" id="{escape(residue.lower())}">
              <div class="panel-header">
                <h2>{escape(residue)} <span>({escape(one_letter)})</span></h2>
                <a href="#top">Back to top</a>
              </div>
              {html}
            </section>
            """
        )

    nav_links = " ".join(f'<a href="#{residue.lower()}">{residue}</a>' for residue, _ in AA_ORDER if residue in residue_values)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BMRB carbon chemical shift charts</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --line: #dbe2ea;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: "DejaVu Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.08), transparent 24rem),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }}
    main {{
      width: min(1500px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    header {{ padding: 10px 0 22px; }}
    h1 {{
      margin: 0 0 8px;
      font-family: "DejaVu Serif", serif;
      font-size: clamp(2rem, 3vw, 2.8rem);
      line-height: 1.05;
    }}
    .lede {{
      max-width: 78ch;
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.5;
    }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 0 16px;
      margin: 18px 0 22px;
      background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(248,250,252,0.88));
      backdrop-filter: blur(6px);
      border-bottom: 1px solid rgba(219,226,234,0.9);
    }}
    nav a {{
      text-decoration: none;
      color: var(--accent);
      border: 1px solid rgba(15,118,110,0.16);
      background: rgba(255,255,255,0.82);
      border-radius: 999px;
      padding: 6px 11px;
      font-size: 0.92rem;
      font-weight: 600;
    }}
    .residue-panel {{
      background: rgba(255,255,255,0.94);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 20px 50px rgba(15, 23, 42, 0.06);
      padding: 18px 18px 8px;
      margin-bottom: 24px;
    }}
    .panel-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      padding: 4px 8px 0;
    }}
    .panel-header h2 {{
      margin: 0;
      font-family: "DejaVu Serif", serif;
      font-size: 1.45rem;
    }}
    .panel-header h2 span {{
      color: var(--muted);
      font-size: 0.92em;
    }}
    .panel-header a {{
      color: var(--muted);
      text-decoration: none;
      font-size: 0.92rem;
    }}
    .plotly-graph-div {{ width: 100%; }}
    @media (max-width: 700px) {{
      main {{ width: min(100vw - 16px, 1500px); }}
      nav {{ gap: 6px; }}
      nav a {{ padding: 5px 9px; font-size: 0.85rem; }}
      .residue-panel {{ padding-left: 8px; padding-right: 8px; }}
      .panel-header {{ padding: 4px 4px 0; }}
    }}
  </style>
</head>
<body>
  <main id="top">
    <header>
      <h1>BMRB Carbon Chemical Shift Charts</h1>
      <p class="lede">Each amino acid gets its own long shared-zoom Plotly chart. Drag to zoom the ppm axis, double-click to reset, and use the legend to isolate helix, sheet, coil, or aggregate traces.</p>
    </header>
    <nav>{nav_links}</nav>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a standalone interactive HTML page for carbon chemical shift charts.")
    parser.add_argument("--filtered-csv", type=Path, default=DATA_CACHE_DIR / "filtered_histogram_cache_compact.csv")
    parser.add_argument("--ss-csv", type=Path, default=DATA_CACHE_DIR / "ss_histogram_cache_compact.csv")
    parser.add_argument(
        "--output-html",
        type=Path,
        default=OUTPUT_DIR / "bmrb_histogram_carbon_charts_interactive.html",
    )
    parser.add_argument("--bin-size", type=float, default=0.25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    filtered_rows = read_filtered_compact(args.filtered_csv)
    ss_rows = read_ss_compact(args.ss_csv)
    write_interactive_html(filtered_rows, ss_rows, args.output_html, args.bin_size)
    print(f"Wrote {args.output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
