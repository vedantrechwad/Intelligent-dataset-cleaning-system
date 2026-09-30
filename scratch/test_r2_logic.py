import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cleaner.io import load_table

# Check beers
df_beers = load_table("benchmarks/beers/dirty.csv")
from cleaner.rules.r2_numeric import parse_numeric_cell

for col in ["ounces", "abv", "ibu"]:
    if col in df_beers.columns:
        parsed = [parse_numeric_cell(v) for v in df_beers[col] if v.strip()]
        parsed = [p for p in parsed if p is not None]
        suffixes = {}
        for p in parsed:
            s = p[2].lower()
            suffixes[s] = suffixes.get(s, 0) + 1
        print(f"Beers {col} suffixes:", suffixes)

# Check movies_1
df_movies = load_table("benchmarks/movies_1/dirty.csv")
for col in ["duration", "rating_value", "rating_count"]:
    if col in df_movies.columns:
        parsed = [parse_numeric_cell(v) for v in df_movies[col] if v.strip()]
        parsed = [p for p in parsed if p is not None]
        suffixes = {}
        for p in parsed:
            s = p[2].lower()
            suffixes[s] = suffixes.get(s, 0) + 1
        print(f"Movies {col} suffixes:", suffixes)
