#!/usr/bin/env python3
"""
Build an archive-wide assignment-level chemical-shift CSV directly from BMRB entry data.

Creation date:
- 2026-04-03

Purpose:
- pull chemical-shift assignment data directly from the BMRB API
- build a master table that preserves enough identity information to trace each
  chemical shift back to its BMRB entry, entity, residue context, and atom
  assignment
- support analyses that need assignment-level identity, including atom-to-atom
  correlation within the same chemical entity

Primary data source:
- BMRB API root: https://api.bmrb.io/v2

Expected inputs:
- BMRB entry JSON data fetched from the BMRB API
- no cached histogram HTML pages are used

Primary outputs:
- `bmrb_data_cache/unfiltered_assignment_level_dataset/bmrb_shift_table_full.csv`
  by default
- a local JSON cache of fetched BMRB entry payloads in `.cache/bmrb_shift_table/`

What the output contains:
- one row per chemical-shift assignment from the BMRB archive
- BMRB entry and entity identity fields
- residue and atom assignment context
- assigned chemical-shift values

What the output does not contain:
- no secondary-structure labels
- no PDB-derived annotations
- no filtering based on the exclusion criteria described on the filtered BMRB
  histogram/statistics page

Filtering note:
- this table is not filtered the way the BMRB filtered histogram/statistics
  tables are filtered
- it does not exclude entries with aromatic or paramagnetic ligands
- it does not exclude entries based on the outlier and unusual-referencing
  criteria described on the filtered statistics page
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

try:
    import certifi  # type: ignore
except ImportError:
    certifi = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BMRB_API_ROOT = "https://api.bmrb.io/v2"
WITHDRAWN_URL = "https://bmrb.io/data_library/withdrawn.shtml"

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def build_ssl_context(cafile: str | None = None) -> ssl.SSLContext:
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def http_get_json(url: str, cafile: str | None = None, max_attempts: int = 5) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "bmrb-shift-table/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, context=build_ssl_context(cafile)) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 15))
                last_error = exc
                continue
            raise RuntimeError(f"HTTP {exc.code} for {url}: {message}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 15))
                last_error = exc
                continue
            raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc
    if last_error is not None:
        raise RuntimeError(f"Failed to fetch {url}: {last_error}")
    raise RuntimeError(f"Failed to fetch {url}")


class JsonCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def set(self, key: str, value: Any) -> None:
        path = self.root / f"{key}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle)


def http_get_text(url: str, cafile: str | None = None, max_attempts: int = 5) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "bmrb-shift-table/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, context=build_ssl_context(cafile)) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 15))
                last_error = exc
                continue
            raise RuntimeError(f"HTTP {exc.code} for {url}: {message}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 15))
                last_error = exc
                continue
            raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc
    if last_error is not None:
        raise RuntimeError(f"Failed to fetch {url}: {last_error}")
    raise RuntimeError(f"Failed to fetch {url}")


def list_entries(cache: JsonCache, cafile: str | None = None) -> list[str]:
    payload = cache.get("list_entries")
    if payload is None:
        payload = http_get_json(f"{BMRB_API_ROOT}/list_entries", cafile=cafile)
        cache.set("list_entries", payload)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected list_entries payload: {type(payload).__name__}")
    return [str(entry_id) for entry_id in payload]


def list_withdrawn_entries(cache: JsonCache, cafile: str | None = None) -> set[str]:
    payload = cache.get("withdrawn_entries")
    if payload is None:
        html = http_get_text(WITHDRAWN_URL, cafile=cafile)
        withdrawn = sorted(set(re.findall(r"(?m)^\\s*(\\d+)\\s+\\d{4}-\\d{2}-\\d{2}\\s*$", html)))
        payload = withdrawn
        cache.set("withdrawn_entries", payload)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected withdrawn_entries payload: {type(payload).__name__}")
    return {str(entry_id) for entry_id in payload}


def fetch_entry_cached(bmrb_id: str, cache: JsonCache, cafile: str | None = None) -> dict[str, Any]:
    key = f"entry_{bmrb_id}"
    payload = cache.get(key)
    if payload is None:
        payload = http_get_json(f"{BMRB_API_ROOT}/entry/{bmrb_id}", cafile=cafile)
        cache.set(key, payload)
    entry = payload.get(str(bmrb_id))
    if not isinstance(entry, dict):
        raise RuntimeError(f"Unexpected entry payload for {bmrb_id}")
    return entry


def iter_atom_shift_rows(entry: dict[str, Any]) -> Iterable[dict[str, str]]:
    for saveframe in entry.get("saveframes", []):
        for loop in saveframe.get("loops", []):
            if loop.get("category") != "_Atom_chem_shift":
                continue
            tags = loop.get("tags", [])
            for row in loop.get("data", []):
                if len(row) == len(tags):
                    yield dict(zip(tags, row))


def process_entry(bmrb_id: str, entry_cache: JsonCache, cafile: str | None = None) -> list[dict[str, Any]]:
    entry = fetch_entry_cached(bmrb_id, entry_cache, cafile=cafile)
    rows: list[dict[str, Any]] = []
    for row in iter_atom_shift_rows(entry):
        residue = row.get("Comp_ID", "").upper()
        if residue not in AA3_TO_1:
            continue
        atom = row.get("Atom_ID", "").upper()
        if not atom:
            continue
        try:
            shift_ppm = float(row["Val"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(
            {
                "bmrb_id": bmrb_id,
                "entity_id": row.get("Entity_ID", ""),
                "entity_assembly_id": row.get("Entity_assembly_ID", ""),
                "comp_index_id": row.get("Comp_index_ID", ""),
                "residue_3": residue,
                "residue_1": AA3_TO_1[residue],
                "atom": atom,
                "element": row.get("Atom_type", "").upper() or atom[:1],
                "shift_ppm": shift_ppm,
                "ambiguity_code": row.get("Ambiguity_code", ""),
                "assigned_chem_shift_list_id": row.get("Assigned_chem_shift_list_ID", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bmrb_id",
                "entity_id",
                "entity_assembly_id",
                "comp_index_id",
                "residue_3",
                "residue_1",
                "atom",
                "element",
                "shift_ppm",
                "ambiguity_code",
                "assigned_chem_shift_list_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an archive-wide BMRB chemical-shift CSV.")
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "bmrb_data_cache" / "unfiltered_assignment_level_dataset" / "bmrb_shift_table_full.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / ".cache" / "bmrb_shift_table"),
        help="Cache directory for BMRB entry fetches.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        help="Only process the first N BMRB entries, for testing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker thread count. Default: 8.",
    )
    parser.add_argument(
        "--cafile",
        default=certifi.where() if certifi is not None else None,
        help="CA bundle path for HTTPS verification.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = Path(args.cache_dir)
    meta_cache = JsonCache(cache_dir / "meta")
    entry_cache = JsonCache(cache_dir / "entry")

    entry_ids = list_entries(meta_cache, cafile=args.cafile)
    withdrawn_ids = list_withdrawn_entries(meta_cache, cafile=args.cafile)
    entry_ids = [entry_id for entry_id in entry_ids if entry_id not in withdrawn_ids]
    if args.max_entries is not None:
        entry_ids = entry_ids[: args.max_entries]

    all_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_entry, bmrb_id, entry_cache, args.cafile): bmrb_id for bmrb_id in entry_ids}
        for index, future in enumerate(as_completed(futures), start=1):
            bmrb_id = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                print(f"[{index}/{len(entry_ids)}] {bmrb_id}: {len(rows)} rows")
            except Exception as exc:
                print(f"[{index}/{len(entry_ids)}] {bmrb_id}: ERROR {exc}")

    deduped = {
        (
            row["bmrb_id"],
            row["entity_id"],
            row["comp_index_id"],
            row["residue_3"],
            row["atom"],
            row["shift_ppm"],
            row["assigned_chem_shift_list_id"],
        ): row
        for row in all_rows
    }
    rows = sorted(
        deduped.values(),
        key=lambda row: (
            row["residue_3"],
            row["atom"],
            row["bmrb_id"],
            row["entity_id"],
            row["comp_index_id"],
            row["shift_ppm"],
        ),
    )

    output_csv = Path(args.csv)
    write_csv(output_csv, rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
