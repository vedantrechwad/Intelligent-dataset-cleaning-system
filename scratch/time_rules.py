import sys, os, time
sys.path.insert(0, ".")
import pandas as pd
from cleaner.rules.r1_missing import propose_r1_missing
from cleaner.rules.r2_numeric import propose_r2_numeric
from cleaner.rules.r3_whitespace_case import propose_r3_whitespace_case
from cleaner.rules.r4_dates import propose_r4_dates
from cleaner.rules.r5_compound_split import propose_r5_compound_split
from cleaner.rules.r6_dependency_repair import propose_r6_dependency_repair
from cleaner.rules.r7_category_variants import propose_r7_category_variants
from cleaner.rules.r8_exact_duplicates import propose_r8_exact_duplicates

df = pd.read_csv("benchmarks/tax/dirty.csv", nrows=10000, dtype=str).fillna("")

rules = [
    ("R1", propose_r1_missing),
    ("R2", propose_r2_numeric),
    ("R3", propose_r3_whitespace_case),
    ("R4", propose_r4_dates),
    ("R5", propose_r5_compound_split),
    ("R6", propose_r6_dependency_repair),
    ("R7", propose_r7_category_variants),
    ("R8", propose_r8_exact_duplicates),
]

touched = set()
for name, fn in rules:
    t0 = time.time()
    props = fn(df, touched_cells=touched)
    print(f"{name}: {time.time()-t0:.2f}s, {len(props)} proposals")
