import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cleaner.io import load_table

dirty = load_table("benchmarks/movies_1/dirty.csv")
clean = load_table("benchmarks/movies_1/clean.csv")

for r in range(len(dirty)):
    d_val = dirty.at[r, "duration"]
    c_val = clean.at[r, "Duration"]
    if "hr." in d_val:
        print(f"Row {r}: dirty='{d_val}' -> clean='{c_val}'")
        if r > 50:
            break
