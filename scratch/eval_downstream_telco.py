import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine
from src.evaluation.downstream_evaluation import evaluate_downstream_ml

# Load Telco dirty, clean, and run our engine
dirty_path = "benchmarks/telco/dirty.csv"
clean_path = "benchmarks/telco/clean.csv"

df_dirty = load_table(dirty_path)
df_clean = load_table(clean_path)

prof = profile_table(df_dirty)
engine = CleaningEngine()
props = engine.generate_proposals(df_dirty, prof)
approved = [p for p in props if p.tier in ["AUTO", "REVIEW"]]
df_cleaned, _ = engine.apply(df_dirty, approved)

# Downstream ML on Churn target
res = evaluate_downstream_ml(df_dirty, df_cleaned, target_column="Churn", random_seed=42)
print("Telco Downstream ML Evaluation:")
print("Task type:", res.get("task_type"))
print("Dirty ML Metrics:", res.get("corrupted_metrics"))
print("Our Cleaned ML Metrics:", res.get("cleaned_metrics"))
print("Delta (Improvement):", res.get("delta"))

# Also evaluate vs Ground Truth clean
res_gt = evaluate_downstream_ml(df_dirty, df_clean, target_column="Churn", random_seed=42)
print("\nGround Truth Clean ML Metrics:", res_gt.get("cleaned_metrics"))
