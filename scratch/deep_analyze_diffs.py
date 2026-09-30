import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine
from eval.run_benchmark import eq

def analyze_differences(dataset_name: str):
    dirty_path = f"benchmarks/{dataset_name}/dirty.csv"
    clean_path = f"benchmarks/{dataset_name}/clean.csv"
    
    df_dirty = load_table(dirty_path)
    df_clean = load_table(clean_path)
    
    prof = profile_table(df_dirty)
    engine = CleaningEngine()
    proposals = engine.generate_proposals(df_dirty, prof)
    approved = [p for p in proposals if p.tier in ["AUTO", "REVIEW"]]
    df_out, diffs = engine.apply(df_dirty, approved)
    
    n_rows = min(len(df_dirty), len(df_clean), len(df_out))
    n_cols = min(len(df_dirty.columns), len(df_clean.columns), len(df_out.columns))
    
    fixed = []
    damaged = []
    wrong = []
    
    for r in range(n_rows):
        for c in range(n_cols):
            d = df_dirty.iat[r, c]
            cl = df_clean.iat[r, c]
            o = df_out.iat[r, c]
            
            is_error = not eq(d, cl)
            if is_error:
                if eq(o, cl):
                    fixed.append((r, c, df_dirty.columns[c], d, o, cl))
                elif not eq(o, d):
                    wrong.append((r, c, df_dirty.columns[c], d, o, cl))
            else:
                if not eq(o, cl):
                    damaged.append((r, c, df_dirty.columns[c], d, o, cl))
                    
    print(f"\n================ {dataset_name.upper()} ================")
    print(f"Fixed: {len(fixed)}, Damaged: {len(damaged)}, Wrong: {len(wrong)}")
    
    if damaged:
        print(f"\nSample Damaged cells (where D == C, but O != C):")
        for r, c, col, d, o, cl in damaged[:10]:
            print(f"  Row {r}, Col '{col}': raw='{d}' -> our_clean='{o}' (ground_truth='{cl}')")
            
    if wrong:
        print(f"\nSample Wrong cells (where D != C, but O != C and O != D):")
        for r, c, col, d, o, cl in wrong[:10]:
            print(f"  Row {r}, Col '{col}': raw='{d}' -> our_clean='{o}' (ground_truth='{cl}')")

for ds in ["beers", "hospital", "movies_1", "telco"]:
    analyze_differences(ds)
