"""
scripts/download_benchmarks.py
Idempotent downloader for benchmark datasets: hospital, flights, beers.
Fetches from https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/<name>/<file>
Fallback: git clone https://github.com/BigDaMa/raha into tempdir.
Asserts expected shapes:
- hospital: 1000x20
- flights: 2376x7
- beers: 2410x11
"""

import os
import sys
import shutil
import tempfile
import urllib.request
import urllib.error
import subprocess
import pandas as pd

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARKS_DIR = os.path.join(WORKSPACE_ROOT, "benchmarks")

EXPECTED_SHAPES = {
    "hospital": (1000, 20),
    "flights": (2376, 7),
    "beers": (2410, 11)
}

DATASETS = ["hospital", "flights", "beers"]
FILES = ["dirty.csv", "clean.csv"]
BASE_URL = "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets"
REPO_URL = "https://github.com/BigDaMa/raha.git"


def download_file(url: str, dest_path: str) -> bool:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        return True
    except Exception as e:
        print(f"Warning: Failed to fetch {url}: {e}", file=sys.stderr)
        return False


def clone_fallback(missing_items: list) -> bool:
    print("Attempting fallback git clone of BigDaMa/raha...")
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_cmd = ["git", "clone", "--depth", "1", REPO_URL, temp_dir]
        res = subprocess.run(clone_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error cloning repository: {res.stderr}", file=sys.stderr)
            return False

        for name, file_name, dest_path in missing_items:
            src_path = os.path.join(temp_dir, "datasets", name, file_name)
            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)
                print(f"Copied from clone: {name}/{file_name}")
            else:
                print(f"Error: {src_path} not found in cloned repo.", file=sys.stderr)
                return False
    return True


def main():
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    missing_to_fetch = []

    for name in DATASETS:
        ds_dir = os.path.join(BENCHMARKS_DIR, name)
        os.makedirs(ds_dir, exist_ok=True)
        for f in FILES:
            dest = os.path.join(ds_dir, f)
            if not os.path.exists(dest) or os.path.getsize(dest) == 0:
                missing_to_fetch.append((name, f, dest))
            else:
                print(f"[EXISTS] {name}/{f} already exists. Skipping download.")

    if missing_to_fetch:
        print(f"Need to download {len(missing_to_fetch)} files...")
        fallback_needed = []
        for name, f, dest in missing_to_fetch:
            url = f"{BASE_URL}/{name}/{f}"
            print(f"Fetching {url} -> {dest}...")
            ok = download_file(url, dest)
            if not ok:
                fallback_needed.append((name, f, dest))

        if fallback_needed:
            success = clone_fallback(fallback_needed)
            if not success:
                print("FATAL: Failed to download all benchmark files via raw URL and git clone fallback.", file=sys.stderr)
                sys.exit(1)

    print("\n--- Verifying Dataset Shapes ---")
    all_ok = True
    for name, expected_shape in EXPECTED_SHAPES.items():
        dirty_path = os.path.join(BENCHMARKS_DIR, name, "dirty.csv")
        clean_path = os.path.join(BENCHMARKS_DIR, name, "clean.csv")

        if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
            print(f"FATAL: Missing files for dataset {name}", file=sys.stderr)
            all_ok = False
            continue

        df_dirty = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)
        df_clean = pd.read_csv(clean_path, dtype=str, keep_default_na=False)

        print(f"Dataset '{name}': dirty.shape={df_dirty.shape}, clean.shape={df_clean.shape}, expected={expected_shape}")

        if df_dirty.shape != df_clean.shape:
            print(f"FATAL: {name} dirty shape {df_dirty.shape} != clean shape {df_clean.shape}", file=sys.stderr)
            all_ok = False
        elif df_dirty.shape != expected_shape:
            print(f"FATAL: {name} shape {df_dirty.shape} != expected shape {expected_shape}", file=sys.stderr)
            all_ok = False

    if not all_ok:
        print("\nFATAL ERROR: Shape validation failed. Stopping as requested.", file=sys.stderr)
        sys.exit(1)

    print("\nSUCCESS: All datasets downloaded/verified with matching shapes.")


if __name__ == "__main__":
    main()
