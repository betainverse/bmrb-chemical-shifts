#!/usr/bin/env python3
"""Convert cached filtered BMRB histogram HTML pages into a compact CSV.

The output is one row per Plotly trace, not one row per individual shift.
That keeps the file compact while preserving:
- page-level residue/atom identity
- trace-level atom naming (for example ALA-HB1/HB2/HB3 on ALA-MB.html)
- the full list of raw chemical shift values as JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bmrb_data_cache.bmrb_ss_histogram_parser import extract_plotly_traces


DEFAULT_CACHE_DIR = PROJECT_ROOT / "bmrb_data_cache" / "bmrb_histogram_pages_filtered_full"
DEFAULT_OUTPUT = DEFAULT_CACHE_DIR / "filtered_histogram_cache_compact.csv"

FILENAME_RE = re.compile(r"histograms_([A-Z0-9]+)-([A-Z0-9]+(?:[0-9]+)?)\.html\.html$")
TRACE_ATOM_RE = re.compile(r"^[A-Z0-9]+-([A-Z0-9]+)$")


def cached_filename_to_page_ids(path: Path) -> tuple[str, str] | None:
    match = FILENAME_RE.search(path.name)
    if not match:
        return None
    return match.group(1), match.group(2)


def trace_name_to_atom(trace_name: str) -> str:
    match = TRACE_ATOM_RE.match(trace_name.strip().upper())
    if match:
        return match.group(1)
    return trace_name.strip().upper()


def convert(cache_dir: Path, output_csv: Path) -> int:
    rows: list[dict[str, object]] = []
    for path in sorted(cache_dir.glob("*.html")):
        page_ids = cached_filename_to_page_ids(path)
        if page_ids is None:
            continue
        residue, page_atom = page_ids
        html = path.read_text(encoding="utf-8")
        traces = extract_plotly_traces(html)
        for index, trace in enumerate(traces, start=1):
            xs = trace.get("x", [])
            if not isinstance(xs, list):
                continue
            values: list[float] = []
            for value in xs:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            if not values:
                continue
            trace_name = str(trace.get("name", "")).strip()
            rows.append(
                {
                    "residue_3": residue,
                    "page_atom": page_atom,
                    "trace_index": index,
                    "trace_name": trace_name,
                    "trace_atom": trace_name_to_atom(trace_name) if trace_name else "",
                    "shift_count": len(values),
                    "min_shift_ppm": min(values),
                    "max_shift_ppm": max(values),
                    "shifts_json": json.dumps(values, separators=(",", ":")),
                    "source_file": path.name,
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "residue_3",
                "page_atom",
                "trace_index",
                "trace_name",
                "trace_atom",
                "shift_count",
                "min_shift_ppm",
                "max_shift_ppm",
                "shifts_json",
                "source_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"cache_dir={cache_dir}")
    print(f"output_csv={output_csv}")
    print(f"rows={len(rows)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert cached filtered BMRB histogram HTML to a compact CSV.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return convert(args.cache_dir, args.output_csv)


if __name__ == "__main__":
    raise SystemExit(main())
