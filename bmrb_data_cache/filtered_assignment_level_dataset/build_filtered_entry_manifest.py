#!/usr/bin/env python3
"""
Created: 2026-04-06

Purpose:
- build an entry-level exclusion manifest for the filtered assignment-level
  dataset workflow
- reuse the same BMRB API-driven entry universe and cache conventions as the
  current unfiltered assignment-level builder
- record both included and excluded BMRB entries, along with best-fit local
  exclusion reasons that approximate the filtered BMRB histogram/statistics
  workflow described at
  `https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt`

Expected input data:
- BMRB entry JSON data fetched from the BMRB API at `https://api.bmrb.io/v2`
- a local entry cache in `.cache/bmrb_shift_table/` when available
- no compact histogram CSVs are required
- only numeric BMRB entry IDs are handled in this first implementation
- non-numeric IDs returned by `list_entries`, such as `bmse000001`, are
  intentionally skipped

Expected output data:
- `bmrb_filtered_entry_manifest.csv` in
  `chem_shifts/bmrb_data_cache/filtered_assignment_level_dataset/`

Relationship to other workflows:
- this script does not rebuild the current unfiltered assignment-level CSV
- this script is the first phase of the future filtered assignment-level
  pipeline
- it depends on the same entry universe and cache shape used by
  `unfiltered_assignment_level_dataset/bmrb_shift_table.py`
- it approximates the exclusion rules described by BMRB at
  `https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt`
- it currently produces manifest rows only for numeric BMRB entry IDs

Command-line use:
- default invocation:
  `python3 bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_entry_manifest.py`
- common options:
  `--entry-cache-dir` to reuse an existing BMRB entry JSON cache
  `--output-csv` to write the manifest to a different location
  `--cafile` to provide an explicit CA bundle for HTTPS requests
  `--max-workers` to control concurrent entry fetches
  `--rule-version` to label the filtering logic used for the build
- example using an existing cache:
  `python3 bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_entry_manifest.py --entry-cache-dir .cache/bmrb_shift_table`
- example writing to a temporary output file:
  `python3 bmrb_data_cache/filtered_assignment_level_dataset/build_filtered_entry_manifest.py --output-csv /tmp/bmrb_filtered_entry_manifest.csv`
- note:
  IDs like `10002` are processed, while non-numeric IDs like `bmse000001`
  are currently skipped
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
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
DEFAULT_RULE_VERSION = "filtered_entry_manifest_v1"

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

AROMATIC_LIGAND_CODES = {
    "ADP",
    "AMP",
    "ATP",
    "FAD",
    "FMN",
    "GDP",
    "GMP",
    "GTP",
    "HEA",
    "HEB",
    "HEC",
    "HEM",
    "NAD",
    "NAP",
    "NDP",
}

PARAMAGNETIC_LIGAND_CODES = {
    "CO",
    "CU",
    "DY",
    "ER",
    "EU",
    "FE",
    "FE2",
    "FE3",
    "GD",
    "MN",
    "NI",
    "TB",
}

AROMATIC_NAME_FRAGMENTS = (
    "ADENOSINE",
    "FLAVIN",
    "HEME",
    "HEMIN",
    "NAD",
    "NICOTINAMIDE",
    "PORPHYRIN",
    "RIBOFLAVIN",
)

PARAMAGNETIC_NAME_FRAGMENTS = (
    "COBALT",
    "COPPER",
    "DYSPROSIUM",
    "ERBIUM",
    "EUROPIUM",
    "GADOLINIUM",
    "IRON",
    "MANGANESE",
    "NICKEL",
    "PARAMAGNETIC",
    "TERBIUM",
)

BACKBONE_NITROGEN_HYDROGENS = {"H", "H1", "H2", "H3", "HN", "HN1", "HN2", "HN3"}

RESIDUE_HETERO_BOUND_HYDROGENS = {
    "ARG": {"HE", "HH11", "HH12", "HH21", "HH22", "HH1", "HH2"},
    "ASN": {"HD21", "HD22"},
    "ASP": {"HD2"},
    "CYS": {"HG"},
    "GLN": {"HE21", "HE22"},
    "GLU": {"HE2"},
    "HIS": {"HD1", "HE2"},
    "LYS": {"HZ", "HZ1", "HZ2", "HZ3"},
    "SER": {"HG"},
    "THR": {"HG1"},
    "TRP": {"HE1"},
    "TYR": {"HH"},
}

REASON_SUMMARIES = {
    "withdrawn_entry": "withdrawn entry",
    "paramagnetic_or_aromatic_ligand": "paramagnetic or aromatic ligand",
    "eight_sigma_outlier": "eight-sigma outlier",
    "carbon_bound_proton_extreme": "carbon-bound proton extreme",
    "insufficient_rule_data": "insufficient rule data",
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
        headers={"Accept": "application/json", "User-Agent": "bmrb-filtered-entry-manifest/1.0"},
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


def http_get_text(url: str, cafile: str | None = None, max_attempts: int = 5) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "bmrb-filtered-entry-manifest/1.0"})
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
        payload = sorted(set(re.findall(r"(?m)^\\s*(\\d+)\\s+\\d{4}-\\d{2}-\\d{2}\\s*$", html)))
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


def saveframe_tag_map(saveframe: dict[str, Any]) -> dict[str, str]:
    tags = saveframe.get("tags") or []
    if isinstance(tags, dict):
        return {str(key): str(value) for key, value in tags.items()}
    if isinstance(tags, list):
        mapped: dict[str, str] = {}
        for item in tags:
            if isinstance(item, list) and len(item) >= 2:
                mapped[str(item[0])] = str(item[1])
        return mapped
    return {}


def iter_atom_shift_rows(entry: dict[str, Any]) -> Iterable[dict[str, str]]:
    for saveframe in entry.get("saveframes", []):
        for loop in saveframe.get("loops", []):
            if loop.get("category") != "_Atom_chem_shift":
                continue
            tags = loop.get("tags", [])
            for row in loop.get("data", []):
                if len(row) == len(tags):
                    yield dict(zip(tags, row))


def safe_positive_count(value: str | None) -> bool | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned in {"", "."}:
        return None
    try:
        return int(cleaned) > 0
    except ValueError:
        return None


def normalize_component_name(raw_value: str) -> str:
    return raw_value.strip().upper()


def component_matches_keywords(value: str, exact_codes: set[str], name_fragments: tuple[str, ...]) -> bool:
    normalized = normalize_component_name(value)
    if normalized in exact_codes:
        return True
    return any(fragment in normalized for fragment in name_fragments)


def classify_ligands(entry: dict[str, Any]) -> tuple[str, str, bool]:
    assembly_maps = [saveframe_tag_map(sf) for sf in entry.get("saveframes", []) if sf.get("category") == "assembly"]
    entity_maps = [saveframe_tag_map(sf) for sf in entry.get("saveframes", []) if sf.get("category") == "entity"]

    organic_flags = [safe_positive_count(tags.get("Organic_ligands")) for tags in assembly_maps]
    metal_flags = [safe_positive_count(tags.get("Metal_ions")) for tags in assembly_maps]
    non_polymer_names = [
        normalize_component_name(tags.get("Name", ""))
        for tags in entity_maps
        if tags.get("Type", "").strip().lower() == "non-polymer" and tags.get("Name", "").strip()
    ]

    has_declared_ligands = any(flag is True for flag in organic_flags + metal_flags)
    has_non_polymer_entities = bool(non_polymer_names)
    has_ligand_evidence = has_declared_ligands or has_non_polymer_entities

    aromatic_or_paramagnetic = any(
        component_matches_keywords(name, AROMATIC_LIGAND_CODES, AROMATIC_NAME_FRAGMENTS)
        or component_matches_keywords(name, PARAMAGNETIC_LIGAND_CODES, PARAMAGNETIC_NAME_FRAGMENTS)
        for name in non_polymer_names
    )

    if has_ligand_evidence:
        ligand_flag = "yes"
    elif any(flag is None for flag in organic_flags + metal_flags) and assembly_maps:
        ligand_flag = "unknown"
    else:
        ligand_flag = "no"

    if aromatic_or_paramagnetic:
        return ligand_flag, "yes", False
    if has_ligand_evidence and not non_polymer_names:
        return ligand_flag, "unknown", True
    if ligand_flag == "unknown":
        return ligand_flag, "unknown", True
    if has_ligand_evidence and non_polymer_names:
        return ligand_flag, "unknown", False
    return ligand_flag, "no", False


def is_carbon_bound_hydrogen(residue: str, atom: str, element: str) -> bool:
    residue = residue.upper()
    atom = atom.upper()
    element = element.upper()
    if element != "H":
        return False
    if atom in BACKBONE_NITROGEN_HYDROGENS:
        return False
    if atom in RESIDUE_HETERO_BOUND_HYDROGENS.get(residue, set()):
        return False
    return atom.startswith("H")


@dataclass
class EntryBaseline:
    bmrb_id: str
    ligand_flag: str
    paramagnetic_or_aromatic_ligand_flag: str
    ligand_unknown: bool
    carbon_bound_proton_extreme_flag: str
    residue_atom_totals: dict[tuple[str, str], tuple[int, float, float]]


@dataclass
class RunningStats:
    count: int = 0
    sum_value: float = 0.0
    sum_squares: float = 0.0

    def add_batch(self, count: int, sum_value: float, sum_squares: float) -> None:
        self.count += count
        self.sum_value += sum_value
        self.sum_squares += sum_squares

    @property
    def mean(self) -> float:
        return self.sum_value / self.count if self.count else 0.0

    @property
    def stddev(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean
        variance = max(0.0, (self.sum_squares / self.count) - (mean * mean))
        return math.sqrt(variance)


def summarize_entry_for_baseline(bmrb_id: str, entry_cache: JsonCache, cafile: str | None = None) -> EntryBaseline:
    entry = fetch_entry_cached(bmrb_id, entry_cache, cafile=cafile)
    ligand_flag, aromatic_flag, ligand_unknown = classify_ligands(entry)

    residue_atom_totals: dict[tuple[str, str], tuple[int, float, float]] = {}
    carbon_bound_extreme = False

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

        key = (residue, atom)
        count, sum_value, sum_squares = residue_atom_totals.get(key, (0, 0.0, 0.0))
        residue_atom_totals[key] = (count + 1, sum_value + shift_ppm, sum_squares + (shift_ppm * shift_ppm))

        element = row.get("Atom_type", "").upper() or atom[:1]
        if is_carbon_bound_hydrogen(residue, atom, element) and (shift_ppm > 10.0 or shift_ppm < -2.5):
            carbon_bound_extreme = True

    return EntryBaseline(
        bmrb_id=bmrb_id,
        ligand_flag=ligand_flag,
        paramagnetic_or_aromatic_ligand_flag=aromatic_flag,
        ligand_unknown=ligand_unknown,
        carbon_bound_proton_extreme_flag="yes" if carbon_bound_extreme else "no",
        residue_atom_totals=residue_atom_totals,
    )


def entry_has_eight_sigma_outlier(
    bmrb_id: str,
    entry_cache: JsonCache,
    stats_by_residue_atom: dict[tuple[str, str], RunningStats],
    cafile: str | None = None,
) -> bool:
    entry = fetch_entry_cached(bmrb_id, entry_cache, cafile=cafile)
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

        stats = stats_by_residue_atom.get((residue, atom))
        if stats is None or stats.count <= 1:
            continue
        stddev = stats.stddev
        if stddev <= 0.0:
            continue
        if abs(shift_ppm - stats.mean) > 8.0 * stddev:
            return True
    return False


def reason_summary(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "included"
    return "; ".join(REASON_SUMMARIES.get(code, code.replace("_", " ")) for code in reason_codes)


def write_manifest_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bmrb_id",
                "manifest_status",
                "exclude_from_filtered_dataset",
                "reason_codes",
                "reason_summary",
                "ligand_flag",
                "paramagnetic_or_aromatic_ligand_flag",
                "eight_sigma_outlier_flag",
                "carbon_bound_proton_extreme_flag",
                "withdrawn_entry_flag",
                "rule_version",
                "built_at",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a filtered BMRB entry exclusion manifest.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "bmrb_data_cache" / "filtered_assignment_level_dataset" / "bmrb_filtered_entry_manifest.csv",
    )
    parser.add_argument(
        "--cafile",
        default=certifi.where() if certifi is not None else None,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--entry-cache-dir",
        type=Path,
        default=REPO_ROOT / ".cache" / "bmrb_shift_table",
    )
    parser.add_argument(
        "--rule-version",
        default=DEFAULT_RULE_VERSION,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = Path(args.entry_cache_dir)
    meta_cache = JsonCache(cache_dir / "meta")
    entry_cache = JsonCache(cache_dir / "entry")

    entry_ids = [entry_id for entry_id in list_entries(meta_cache, cafile=args.cafile) if entry_id.isdigit()]
    withdrawn_ids = list_withdrawn_entries(meta_cache, cafile=args.cafile)

    baseline_by_entry: dict[str, EntryBaseline] = {}
    stats_by_residue_atom: dict[tuple[str, str], RunningStats] = {}

    print(f"Building baseline summaries for {len(entry_ids)} entries")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(summarize_entry_for_baseline, bmrb_id, entry_cache, args.cafile): bmrb_id
            for bmrb_id in entry_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            bmrb_id = futures[future]
            baseline = future.result()
            baseline_by_entry[bmrb_id] = baseline
            for key, (count, sum_value, sum_squares) in baseline.residue_atom_totals.items():
                stats_by_residue_atom.setdefault(key, RunningStats()).add_batch(count, sum_value, sum_squares)
            if index % 500 == 0 or index == len(entry_ids):
                print(f"[baseline {index}/{len(entry_ids)}]")

    outlier_flag_by_entry: dict[str, str] = {}
    print(f"Scanning for eight-sigma outliers across {len(entry_ids)} entries")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(entry_has_eight_sigma_outlier, bmrb_id, entry_cache, stats_by_residue_atom, args.cafile): bmrb_id
            for bmrb_id in entry_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            bmrb_id = futures[future]
            outlier_flag_by_entry[bmrb_id] = "yes" if future.result() else "no"
            if index % 500 == 0 or index == len(entry_ids):
                print(f"[outliers {index}/{len(entry_ids)}]")

    built_at = datetime.now().isoformat(timespec="seconds")
    manifest_rows: list[dict[str, str]] = []
    for bmrb_id in sorted(entry_ids, key=lambda value: int(value)):
        baseline = baseline_by_entry[bmrb_id]
        reason_codes: list[str] = []

        withdrawn_flag = "yes" if bmrb_id in withdrawn_ids else "no"
        if withdrawn_flag == "yes":
            reason_codes.append("withdrawn_entry")
        if baseline.paramagnetic_or_aromatic_ligand_flag == "yes":
            reason_codes.append("paramagnetic_or_aromatic_ligand")
        if outlier_flag_by_entry.get(bmrb_id, "no") == "yes":
            reason_codes.append("eight_sigma_outlier")
        if baseline.carbon_bound_proton_extreme_flag == "yes":
            reason_codes.append("carbon_bound_proton_extreme")

        manifest_status = "excluded" if reason_codes else "included"
        manifest_rows.append(
            {
                "bmrb_id": bmrb_id,
                "manifest_status": manifest_status,
                "exclude_from_filtered_dataset": "yes" if manifest_status == "excluded" else "no",
                "reason_codes": ";".join(reason_codes),
                "reason_summary": reason_summary(reason_codes),
                "ligand_flag": baseline.ligand_flag,
                "paramagnetic_or_aromatic_ligand_flag": baseline.paramagnetic_or_aromatic_ligand_flag,
                "eight_sigma_outlier_flag": outlier_flag_by_entry.get(bmrb_id, "no"),
                "carbon_bound_proton_extreme_flag": baseline.carbon_bound_proton_extreme_flag,
                "withdrawn_entry_flag": withdrawn_flag,
                "rule_version": args.rule_version,
                "built_at": built_at,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(args.output_csv, manifest_rows)
    print(f"Wrote {len(manifest_rows)} manifest rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
