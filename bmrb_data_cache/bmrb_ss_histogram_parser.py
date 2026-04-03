#!/usr/bin/env python3
"""
Extract BMRB secondary-structure histogram data from a Plotly `_ss.html` page.

These BMRB pages embed raw per-trace chemical shift values directly in the HTML.
This script pulls those values out so we can inspect or reuse BMRB's own
secondary-structure categories without rebuilding them from PDB mappings.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import certifi  # type: ignore
except ImportError:
    certifi = None


DEFAULT_URL_TEMPLATE = "https://bmrb.io/ref_info/histograms/{residue}-{atom}_ss.html"
USER_AGENT = "bmrb-ss-histogram-parser/1.0"


def build_ssl_context(cafile: str | None = None, insecure: bool = False) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def fetch_text(url: str, cafile: str | None = None, insecure: bool = False) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, context=build_ssl_context(cafile, insecure)) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_plotly_traces(html: str) -> list[dict[str, Any]]:
    start_call = html.rfind("Plotly.newPlot(")
    if start_call == -1:
        raise RuntimeError("Could not find Plotly.newPlot(...) in HTML")

    start = html.find("[", start_call)
    if start == -1:
        raise RuntimeError("Could not locate Plotly trace array")

    decoder = json.JSONDecoder()
    traces, end = decoder.raw_decode(html[start:])
    if not isinstance(traces, list):
        raise RuntimeError("Unexpected Plotly trace payload")
    _ = end
    return traces


def residue_atom_from_url(url: str) -> tuple[str, str]:
    name = url.rsplit("/", 1)[-1]
    stem = name.removesuffix(".html")
    if stem.endswith("_ss"):
        stem = stem[:-3]
    residue, atom = stem.split("-", 1)
    return residue.upper(), atom.upper()


def normalize_state(name: str) -> str:
    value = name.upper()
    mapping = {
        "HELX_P": "helix",
        "SHEET": "sheet",
        "COIL": "coil",
        "TURN_P": "turn",
    }
    return mapping.get(value, value.lower())


def traces_to_rows(traces: list[dict[str, Any]], residue: str, atom: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        state_raw = str(trace.get("name", "")).strip()
        if not state_raw:
            continue
        xs = trace.get("x", [])
        if not isinstance(xs, list):
            continue
        state = normalize_state(state_raw)
        for value in xs:
            try:
                shift = float(value)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "residue_3": residue,
                    "atom": atom,
                    "secondary_structure_raw": state_raw,
                    "secondary_structure": state,
                    "shift_ppm": shift,
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["secondary_structure_raw"] for row in rows)
    summary: list[dict[str, Any]] = []
    for raw_state, count in sorted(counts.items()):
        shifts = [float(row["shift_ppm"]) for row in rows if row["secondary_structure_raw"] == raw_state]
        summary.append(
            {
                "secondary_structure_raw": raw_state,
                "secondary_structure": normalize_state(raw_state),
                "count": count,
                "min_shift_ppm": min(shifts),
                "max_shift_ppm": max(shifts),
            }
        )
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "residue_3",
                "atom",
                "secondary_structure_raw",
                "secondary_structure",
                "shift_ppm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "secondary_structure_raw",
                "secondary_structure",
                "count",
                "min_shift_ppm",
                "max_shift_ppm",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a BMRB secondary-structure histogram page.")
    parser.add_argument("residue", nargs="?", help="Three-letter residue code, e.g. ALA")
    parser.add_argument("atom", nargs="?", help="Atom name, e.g. H or CA")
    parser.add_argument(
        "--url",
        help="Explicit BMRB secondary-structure histogram URL. Overrides residue/atom.",
    )
    parser.add_argument("--output-csv", type=Path, help="Write one row per chemical shift value.")
    parser.add_argument("--summary-csv", type=Path, help="Write counts/min/max per secondary-structure class.")
    parser.add_argument("--cafile", help="Custom CA bundle path.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification.")
    args = parser.parse_args()

    if args.url:
        url = args.url
        residue, atom = residue_atom_from_url(url)
    else:
        if not args.residue or not args.atom:
            parser.error("Provide residue and atom, or use --url.")
        residue = args.residue.upper()
        atom = args.atom.upper()
        url = DEFAULT_URL_TEMPLATE.format(residue=residue, atom=atom)

    html = fetch_text(url, cafile=args.cafile, insecure=args.insecure)
    traces = extract_plotly_traces(html)
    rows = traces_to_rows(traces, residue, atom)
    if not rows:
        raise RuntimeError(f"No chemical shift rows found in {url}")
    summary = summarize_rows(rows)

    print(f"url={url}")
    print(f"rows={len(rows)}")
    print("states=" + ", ".join(f"{item['secondary_structure_raw']}:{item['count']}" for item in summary))

    if args.output_csv:
        write_rows_csv(args.output_csv, rows)
    if args.summary_csv:
        write_summary_csv(args.summary_csv, summary)


if __name__ == "__main__":
    main()
