#!/usr/bin/env python3
"""
Bulk-download BMRB secondary-structure histogram data from published `_ss.html`
pages and combine the embedded Plotly traces into one CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bmrb_data_cache.bmrb_ss_histogram_parser import (
    DEFAULT_URL_TEMPLATE,
    fetch_text,
    extract_plotly_traces,
    residue_atom_from_url,
    summarize_rows,
    traces_to_rows,
    write_rows_csv,
    write_summary_csv,
)

DATA_CACHE_DIR = Path(__file__).resolve().parent


DEFAULT_INDEX_URL = "https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt"


def extract_histogram_links(index_html: str, index_url: str) -> list[str]:
    urls = {
        urljoin(index_url, match)
        for match in re.findall(r'href=["\']([^"\']+_ss\.html)["\']', index_html, flags=re.I)
    }
    return sorted(urls)


def parse_include_filters(values: Iterable[str]) -> set[tuple[str, str]]:
    filters: set[tuple[str, str]] = set()
    for value in values:
        residue, atom = value.split(":", 1)
        filters.add((residue.upper(), atom.upper()))
    return filters


def filter_links(urls: list[str], include: set[tuple[str, str]] | None = None) -> list[str]:
    if not include:
        return urls
    filtered: list[str] = []
    for url in urls:
        residue, atom = residue_atom_from_url(url)
        if (residue, atom) in include:
            filtered.append(url)
    return filtered


def collect_rows(
    urls: list[str],
    cafile: str | None = None,
    insecure: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for idx, url in enumerate(urls, start=1):
        residue, atom = residue_atom_from_url(url)
        try:
            html = fetch_text(url, cafile=cafile, insecure=insecure)
            traces = extract_plotly_traces(html)
            rows = traces_to_rows(traces, residue, atom)
            all_rows.extend(rows)
            manifest.append(
                {
                    "url": url,
                    "residue_3": residue,
                    "atom": atom,
                    "status": "ok",
                    "row_count": len(rows),
                    "states": ";".join(
                        f"{item['secondary_structure_raw']}:{item['count']}"
                        for item in summarize_rows(rows)
                    ),
                }
            )
            print(f"[{idx}/{len(urls)}] {residue}-{atom}: {len(rows)} rows", flush=True)
        except Exception as exc:
            manifest.append(
                {
                    "url": url,
                    "residue_3": residue,
                    "atom": atom,
                    "status": "error",
                    "row_count": 0,
                    "states": str(exc),
                }
            )
            print(f"[{idx}/{len(urls)}] {residue}-{atom}: ERROR {exc}", flush=True)
    return all_rows, manifest


def write_manifest_csv(path: Path, manifest: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["url", "residue_3", "atom", "status", "row_count", "states"],
        )
        writer.writeheader()
        writer.writerows(manifest)


def write_global_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["residue_3"]), str(row["atom"]))
        grouped.setdefault(key, []).append(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "residue_3",
                "atom",
                "secondary_structure_raw",
                "secondary_structure",
                "count",
                "min_shift_ppm",
                "max_shift_ppm",
            ],
        )
        writer.writeheader()
        for (residue, atom) in sorted(grouped):
            for item in summarize_rows(grouped[(residue, atom)]):
                writer.writerow(
                    {
                        "residue_3": residue,
                        "atom": atom,
                        **item,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-download BMRB secondary-structure histogram data."
    )
    parser.add_argument(
        "--index-url",
        default=DEFAULT_INDEX_URL,
        help="BMRB stats page to scrape for _ss.html links.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_CACHE_DIR / "bmrb_ss_histograms.csv",
        help="Combined per-shift output CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DATA_CACHE_DIR / "bmrb_ss_histograms_summary.csv",
        help="Combined per-residue/atom summary CSV.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=DATA_CACHE_DIR / "bmrb_ss_histograms_manifest.csv",
        help="Download manifest with row counts and errors.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="RES:ATOM",
        help="Restrict download to specific residue/atom pairs, e.g. ALA:H",
    )
    parser.add_argument("--limit", type=int, help="Only download the first N matched pages.")
    parser.add_argument("--cafile", help="Custom CA bundle path.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification.")
    args = parser.parse_args()

    include = parse_include_filters(args.include) if args.include else None
    index_html = fetch_text(args.index_url, cafile=args.cafile, insecure=args.insecure)
    urls = extract_histogram_links(index_html, args.index_url)
    urls = filter_links(urls, include=include)
    if args.limit is not None:
        urls = urls[: args.limit]

    if not urls:
        raise RuntimeError("No _ss.html links matched the requested filters.")

    print(f"index_url={args.index_url}")
    print(f"pages={len(urls)}")

    rows, manifest = collect_rows(urls, cafile=args.cafile, insecure=args.insecure)
    ok_pages = sum(1 for item in manifest if item["status"] == "ok")
    error_pages = len(manifest) - ok_pages
    state_counts = Counter(str(row["secondary_structure_raw"]) for row in rows)

    print(f"downloaded_pages={ok_pages}")
    print(f"errors={error_pages}")
    print(f"rows={len(rows)}")
    print("states=" + ", ".join(f"{state}:{count}" for state, count in sorted(state_counts.items())))

    write_rows_csv(args.output_csv, rows)
    write_global_summary_csv(args.summary_csv, rows)
    write_manifest_csv(args.manifest_csv, manifest)


if __name__ == "__main__":
    main()
