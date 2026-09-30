import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cleaner.io import load_table

dirty = load_table("benchmarks/movies_1/dirty.csv")
clean = load_table("benchmarks/movies_1/clean.csv")

print("dirty.columns:", list(dirty.columns))
print("clean.columns:", list(clean.columns))

# Print duration column values
for r in range(20):
    d = dirty.iat[r, 10]
    c = clean.iat[r, 10]
    print(f"Row {r:2d} (col 10): dirty='{d}' | clean='{c}'")
