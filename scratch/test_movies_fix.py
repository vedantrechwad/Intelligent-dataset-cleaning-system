import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cleaner.io import load_table
from cleaner.engine import CleaningEngine

# Test R2 proposal generation on movies_1
dirty = load_table("benchmarks/movies_1/dirty.csv")
clean = load_table("benchmarks/movies_1/clean.csv")

engine = CleaningEngine()
props = engine.generate_proposals(dirty)

print("Number of proposals on movies_1:", len(props))
for p in props:
    print(f"  {p.id} [{p.tier}]: {p.description}")
