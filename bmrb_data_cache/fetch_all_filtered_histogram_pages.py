#!/usr/bin/env python3
"""Fetch all published filtered BMRB histogram pages into a fresh cache.

This downloads every non-SS, non-unfiltered histogram page linked from the
BMRB filtered chemical shift statistics index and stores them in a dedicated
filtered-only cache directory.
"""

from __future__ import annotations

from pathlib import Path
import sys
from urllib.error import HTTPError, URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bmrb_data_cache.histogram_cache_utils import HtmlCache, INDEX_URL, extract_histogram_links, fetch_cached_text
from bmrb_data_cache.bmrb_ss_histogram_parser import fetch_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "bmrb_data_cache" / "bmrb_histogram_pages_filtered_full"


def main() -> int:
    cache = HtmlCache(CACHE_DIR)
    index_html = fetch_cached_text(cache, INDEX_URL, None, True)
    links = extract_histogram_links(index_html, INDEX_URL)

    added: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    for url in links:
        if url.endswith("_ss.html") or url.endswith("_unfiltered.html"):
            continue
        if cache.path_for_url(url).exists():
            skipped.append(url)
            continue
        try:
            text = fetch_text(url, insecure=True)
        except (HTTPError, URLError):
            missing.append(url)
            continue
        cache.set(url, text)
        added.append(url)

    print(f"cache_dir={CACHE_DIR}")
    print(f"added={len(added)}")
    print(f"skipped_existing={len(skipped)}")
    print(f"missing={len(missing)}")
    for url in added:
        print(url)
    for url in missing:
        print(f"MISSING {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
