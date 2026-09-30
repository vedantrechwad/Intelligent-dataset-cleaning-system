import os
import urllib.request
import pandas as pd

benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")

def download_dataset(name: str):
    target_dir = os.path.join(benchmarks_dir, name)
    os.makedirs(target_dir, exist_ok=True)
    
    for fname in ["dirty.csv", "clean.csv"]:
        fpath = os.path.join(target_dir, fname)
        if not os.path.exists(fpath):
            url = f"https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/{name}/{fname}"
            print(f"Downloading {url} to {fpath}...")
            urllib.request.urlretrieve(url, fpath)
            print(f"Downloaded {fname}: {os.path.getsize(fpath)} bytes")

download_dataset("tax")

dirty = pd.read_csv(os.path.join(benchmarks_dir, "tax", "dirty.csv"), dtype=str, keep_default_na=False)
clean = pd.read_csv(os.path.join(benchmarks_dir, "tax", "clean.csv"), dtype=str, keep_default_na=False)
print("Tax shape:", dirty.shape)
print("Columns:", list(dirty.columns))

diff_cells = 0
for r in range(min(len(dirty), len(clean))):
    for c in range(min(len(dirty.columns), len(clean.columns))):
        if str(dirty.iloc[r, c]) != str(clean.iloc[r, c]):
            diff_cells += 1
print("Tax error cells (|E|):", diff_cells)
