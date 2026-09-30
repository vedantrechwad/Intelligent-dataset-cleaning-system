import sys, os, time
sys.path.insert(0, ".")
import pandas as pd
from eval.compare_published import run_pipeline_on_dataset, calculate_metrics

t0 = time.time()
print("Starting tax pipeline run...")
df_dirty = pd.read_csv("benchmarks/tax/dirty.csv", dtype=str).fillna("")
df_clean = pd.read_csv("benchmarks/tax/clean.csv", dtype=str).fillna("")

# Let's test on 10,000 rows first to measure throughput
df_d_10k = df_dirty.iloc[:10000].copy()
df_c_10k = df_clean.iloc[:10000].copy()

from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine

prof = profile_table(df_d_10k)
engine = CleaningEngine()
props = engine.generate_proposals(df_d_10k, prof)
approved = [p for p in props if p.tier in ["AUTO", "REVIEW"]]
df_out, diffs = engine.apply(df_d_10k, approved)

m_exact = calculate_metrics(df_d_10k, df_c_10k, df_out, numeric_tolerant=False)
m_num = calculate_metrics(df_d_10k, df_c_10k, df_out, numeric_tolerant=True)

print("10k rows results:")
print(f"  Exact: P={m_exact['precision']:.4f}, R={m_exact['recall']:.4f}, F1={m_exact['f1']:.4f}, Fixed={m_exact['fixed']}/{m_exact['E_count']}, Damaged={m_exact['damaged']}")
print(f"  Numeric: P={m_num['precision']:.4f}, R={m_num['recall']:.4f}, F1={m_num['f1']:.4f}, Fixed={m_num['fixed']}/{m_num['E_count']}, Damaged={m_num['damaged']}")
print(f"Total time: {time.time()-t0:.2f}s")
