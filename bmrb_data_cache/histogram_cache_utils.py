#!/usr/bin/env python3
"""
Created: 2026-04-03

Purpose:
- hold the small shared utility layer for the BMRB histogram-page cache
  workflow
- support standalone scripts that fetch filtered histogram HTML pages, fetch
  secondary-structure histogram HTML pages, and convert cached SS pages into a
  compact CSV
- keep these cache-oriented helpers separate from older chart-generation
  workflows so the data pipeline is easier to understand

Expected input data:
- the filtered BMRB chemical-shift statistics index at:
  https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt
- cached histogram HTML pages stored under `chem_shifts/bmrb_data_cache/`

Expected output data:
- cache files written by the fetch scripts that use `HtmlCache`
- normalized histogram-page URLs and normalized secondary-structure labels used
  by downstream CSV-building scripts

What relies on it:
- `chem_shifts/bmrb_data_cache/fetch_all_filtered_histogram_pages.py`
- `chem_shifts/bmrb_data_cache/fetch_all_ss_histogram_pages.py`
- `chem_shifts/bmrb_data_cache/cached_html_to_csv_ss.py`

Notes:
- this module does not generate charts
- this module exists so active cache scripts do not depend on the deprecated
  `bmrb_histogram_charts.py` workflow
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

from bmrb_data_cache.bmrb_ss_histogram_parser import fetch_text


INDEX_URL = "https://bmrb.io/ref_info/csstats.php?restype=aa&set=filt"


class HtmlCache:
    """Simple URL-to-file cache for locally storing fetched histogram pages."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for_url(self, url: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", url)
        return self.root / f"{safe}.html"

    def get(self, url: str) -> str | None:
        path = self.path_for_url(url)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def set(self, url: str, text: str) -> None:
        self.path_for_url(url).write_text(text, encoding="utf-8")


def fetch_cached_text(cache: HtmlCache, url: str, cafile: str | None, insecure: bool) -> str:
    """Fetch text once, then store it in the provided HTML cache."""

    cached = cache.get(url)
    if cached is not None:
        return cached
    text = fetch_text(url, cafile=cafile, insecure=insecure)
    cache.set(url, text)
    return text


def extract_histogram_links(index_html: str, index_url: str) -> list[str]:
    """Return absolute histogram-page links from the filtered BMRB index page."""

    links = {
        urljoin(index_url, match)
        for match in re.findall(r'href=["\']([^"\']+\.html)["\']', index_html, flags=re.I)
        if "histograms/" in match
    }
    return sorted(links)


def normalize_ss_state(raw_name: str) -> str:
    """Map BMRB trace labels onto the simplified SS labels used in this repo."""

    value = raw_name.upper()
    if value == "HELX_P":
        return "helix"
    if value == "SHEET":
        return "sheet"
    if value in {"COIL", "TURN_P"}:
        return "coil"
    return "other"
