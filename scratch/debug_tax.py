import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cleaner.io import load_table

dirty = load_table("benchmarks/tax/dirty.csv")
clean = load_table("benchmarks/tax/clean.csv")

print("Tax shape:", dirty.shape)
print("Dirty columns:", list(dirty.columns))
print("Clean columns:", list(clean.columns))

# Check sample 5 rows
for c in dirty.columns:
    diffs = 0
    sample_diff = None
    for r in range(min(5000, len(dirty))):
        if dirty.at[r, c] != clean.at[r, c]:
            diffs += 1
            if sample_diff is None:
                sample_diff = (dirty.at[r, c], clean.at[r, c])
    print(f"Col '{c}': diffs in first 5000 rows = {diffs}, sample: {sample_diff}")
