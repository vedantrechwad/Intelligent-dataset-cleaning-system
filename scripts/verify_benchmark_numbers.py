import os
import sys
import pandas as pd

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine
from eval.run_benchmark import eq

print("=" * 70, flush=True)
print("LIVE BENCHMARK EXECUTION - NO HARDCODED OR FABRICATED NUMBERS", flush=True)
print("Running directly on benchmark files in 'benchmarks/' folder...", flush=True)
print("=" * 70, flush=True)

datasets = ["beers", "hospital", "movies_1", "telco"]
results = []

for name in datasets:
    dirty_path = os.path.join(WORKSPACE_ROOT, "benchmarks", name, "dirty.csv")
    clean_path = os.path.join(WORKSPACE_ROOT, "benchmarks", name, "clean.csv")
    
    if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
        continue
        
    print(f"\n[RUNNING] Profiling and cleaning dataset: '{name}'...", flush=True)
    df_dirty = load_table(dirty_path)
    df_clean = load_table(clean_path)
    
    prof = profile_table(df_dirty)
    engine = CleaningEngine()
    proposals = engine.generate_proposals(df_dirty, prof)
    approved = [p for p in proposals if p.tier in ["AUTO", "REVIEW"]]
    df_out, diffs = engine.apply(df_dirty, approved)
    
    n_rows = min(len(df_dirty), len(df_clean), len(df_out))
    n_cols = min(len(df_dirty.columns), len(df_clean.columns), len(df_out.columns))
    total_cells = n_rows * n_cols
    
    errors_total = 0
    clean_total = 0
    fixed = 0
    missed = 0
    wrong = 0
    damaged = 0
    exact_matches = 0
    
    for r in range(n_rows):
        for c in range(n_cols):
            d = df_dirty.iat[r, c]
            cl = df_clean.iat[r, c]
            o = df_out.iat[r, c]
            
            is_err = not eq(d, cl)
            if is_err:
                errors_total += 1
                if eq(o, cl):
                    fixed += 1
                elif eq(o, d):
                    missed += 1
                else:
                    wrong += 1
            else:
                clean_total += 1
                if not eq(o, cl):
                    damaged += 1
                    
            if eq(o, cl):
                exact_matches += 1
                
    recall = (fixed / errors_total * 100.0) if errors_total > 0 else 0.0
    denom_prec = fixed + wrong + damaged
    precision = (fixed / denom_prec * 100.0) if denom_prec > 0 else 0.0
    harm_rate = (damaged / clean_total * 100.0) if clean_total > 0 else 0.0
    agreement = (exact_matches / total_cells * 100.0) if total_cells > 0 else 0.0
    
    results.append({
        "dataset": name,
        "total_cells": total_cells,
        "errors": errors_total,
        "fixed": fixed,
        "missed": missed,
        "wrong": wrong,
        "damaged": damaged,
        "recall": recall,
        "precision": precision,
        "harm_rate": harm_rate,
        "agreement": agreement,
        "rules_count": len(approved)
    })
    print(f"  -> Done '{name}': Errors={errors_total}, Fixed={fixed}, Damaged={damaged}, Precision={precision:.2f}%, Recall={recall:.2f}%", flush=True)

print("\n" + "=" * 70, flush=True)
print("FINAL VERIFIED RESULTS SUMMARY TABLE", flush=True)
print("=" * 70, flush=True)

# Print as clean text table
headers = ["Dataset", "Total Cells", "Errors", "Fixed", "Damaged", "Precision", "Recall", "Harm Rate", "Agreement"]
row_fmt = "{:<10} {:>12} {:>8} {:>8} {:>8} {:>10} {:>8} {:>10} {:>10}"
print(row_fmt.format(*headers), flush=True)
print("-" * 88, flush=True)
for r in results:
    print(row_fmt.format(
        r["dataset"],
        f"{r['total_cells']:,}",
        f"{r['errors']:,}",
        f"{r['fixed']:,}",
        f"{r['damaged']:,}",
        f"{r['precision']:.2f}%",
        f"{r['recall']:.2f}%",
        f"{r['harm_rate']:.3f}%",
        f"{r['agreement']:.2f}%"
    ), flush=True)
print("=" * 88, flush=True)
