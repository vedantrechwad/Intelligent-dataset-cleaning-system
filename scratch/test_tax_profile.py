import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine

t0 = time.time()
df = load_table("benchmarks/tax/dirty.csv")
print(f"Loaded tax in {time.time()-t0:.2f}s, shape: {df.shape}")

# Profile on first 20,000 rows or full
t1 = time.time()
profile = profile_table(df.iloc[:20000])
print(f"Profiled 20k rows in {time.time()-t1:.2f}s")
print(f"Found FDs: {profile.get('candidate_dependencies')}")

engine = CleaningEngine()
t2 = time.time()
props = engine.generate_proposals(df.iloc[:20000], profile)
print(f"Generated proposals in {time.time()-t2:.2f}s: {len(props)} proposals")
for p in props:
    print(f"  {p.id} [{p.tier}]: {p.description}")
