"""
scratch/run_full_neuro_symbolic_benchmarks.py
Comprehensive Empirical Evaluation across Academic, Real-World, and New Kaggle Datasets.
Measures:
- Ingestion & Profiling time
- Rule proposals by tier (AUTO, REVIEW, FLAG)
- Exact cell change counts
- Academic ground truth metrics (Precision, Recall, F1, Harm Rate) where clean.csv exists
- Downstream ML Utility Gains (Accuracy, F1, Precision, Recall, Delta %, Lift) on dirty vs cleaned
"""

import os
import sys
import time
import json
import psutil
import pandas as pd
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine
from eval.compare_published import calculate_metrics
from src.evaluation.downstream_evaluation import evaluate_downstream_ml, infer_target_column

ALL_DATASETS = [
    # 1. Academic Benchmarks (CleanML / Baran ground truth)
    {
        "name": "hospital",
        "category": "Academic Benchmark",
        "domain": "Healthcare Operations",
        "path": "benchmarks/hospital/dirty.csv",
        "clean_path": "benchmarks/hospital/clean.csv",
        "has_ground_truth": True,
        "has_index_col": True,
        "target_col": "HospitalName"
    },
    {
        "name": "beers",
        "category": "Academic Benchmark",
        "domain": "Food & Beverage / Brewing",
        "path": "benchmarks/beers/dirty.csv",
        "clean_path": "benchmarks/beers/clean.csv",
        "has_ground_truth": True,
        "has_index_col": True,
        "target_col": "style"
    },
    {
        "name": "flights",
        "category": "Academic Benchmark",
        "domain": "Aviation Logistics",
        "path": "benchmarks/flights/dirty.csv",
        "clean_path": "benchmarks/flights/clean.csv",
        "has_ground_truth": True,
        "target_col": "flight"
    },
    {
        "name": "movies_1",
        "category": "Academic Benchmark",
        "domain": "Entertainment & Media",
        "path": "benchmarks/movies_1/dirty.csv",
        "clean_path": "benchmarks/movies_1/clean.csv",
        "has_ground_truth": True,
        "target_col": "rating_value"
    },
    {
        "name": "telco",
        "category": "Academic Benchmark",
        "domain": "Telecommunications Churn",
        "path": "benchmarks/telco/dirty.csv",
        "clean_path": "benchmarks/telco/clean.csv",
        "has_ground_truth": True,
        "target_col": "Churn"
    },
    {
        "name": "tax_sample_25k",
        "category": "Academic Benchmark",
        "domain": "IRS Tax Geography",
        "path": "benchmarks/tax/dirty.csv",
        "clean_path": "benchmarks/tax/clean.csv",
        "has_ground_truth": True,
        "max_rows": 25000,
        "target_col": "State"
    },

    # 2. Real-World Benchmarks (Suite 1)
    {
        "name": "adult_census_income (32.5k)",
        "category": "Real-World Benchmark",
        "domain": "Socioeconomic Census",
        "path": "new test datasets/adult (1)/adult.data",
        "header": None,
        "names": [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ],
        "target_col": "income"
    },
    {
        "name": "bank_marketing (Full 45k)",
        "category": "Real-World Benchmark",
        "domain": "Banking Direct Campaigns",
        "path": "new test datasets/bank+marketing/bank-full.csv",
        "target_col": "y"
    },
    {
        "name": "wdbc (Breast Cancer Wisconsin)",
        "category": "Real-World Benchmark",
        "domain": "Medical Oncology",
        "path": "new test datasets/breast+cancer+wisconsin+diagnostic/wdbc.data",
        "header": None,
        "names": ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)],
        "target_col": "diagnosis"
    },
    {
        "name": "student_math (Student Performance)",
        "category": "Real-World Benchmark",
        "domain": "Educational Outcomes",
        "path": "new test datasets/student+performance/student-mat.csv",
        "target_col": "G3"
    },
    {
        "name": "student_portuguese",
        "category": "Real-World Benchmark",
        "domain": "Educational Outcomes",
        "path": "new test datasets/student+performance/student-por.csv",
        "target_col": "G3"
    },
    {
        "name": "credit_card_fraud (284k)",
        "category": "Real-World Benchmark",
        "domain": "Financial Fraud",
        "path": "new test datasets/creditcard.csv/creditcard.csv",
        "max_rows": 50000,
        "target_col": "Class"
    },
    {
        "name": "online_retail_sample_50k",
        "category": "Real-World Benchmark",
        "domain": "E-Commerce Transactions",
        "path": "new test datasets/online+retail/Online Retail.xlsx",
        "is_excel": True,
        "max_rows": 50000,
        "target_col": "Country"
    },

    # 3. New Kaggle Benchmark Suite ("yet another test datasets")
    {
        "name": "airbnb_nyc_2019 (48.8k)",
        "category": "New Kaggle Dataset",
        "domain": "Hospitality & Real Estate",
        "path": "yet another test datasets/AB_NYC_2019.csv/AB_NYC_2019.csv",
        "target_col": "room_type"
    },
    {
        "name": "netflix_titles (8.8k)",
        "category": "New Kaggle Dataset",
        "domain": "Streaming Catalog",
        "path": "yet another test datasets/archive (5)/netflix_titles.csv",
        "target_col": "type"
    },
    {
        "name": "imdb_top_1000",
        "category": "New Kaggle Dataset",
        "domain": "Cinema & Entertainment",
        "path": "yet another test datasets/archive (6)/imdb_top_1000.csv",
        "target_col": "Certificate"
    },
    {
        "name": "world_happiness_2015",
        "category": "New Kaggle Dataset",
        "domain": "Macroeconomics & Quality of Life",
        "path": "yet another test datasets/archive (8)/2015.csv",
        "target_col": "Region"
    },
    {
        "name": "uber_foil_nyc",
        "category": "New Kaggle Dataset",
        "domain": "Urban Transportation Logistics",
        "path": "yet another test datasets/archive (4)/Uber-Jan-Feb-FOIL.csv",
        "target_col": "trips"
    },
    {
        "name": "global_superstore (51.2k)",
        "category": "New Kaggle Dataset",
        "domain": "Global Retail & Logistics",
        "path": "yet another test datasets/archive (7)/Global Superstore.xls",
        "is_excel": True,
        "max_rows": 10000,
        "target_col": "Segment"
    }
]

def run_all():
    print("=" * 85)
    print("STARTING FULL NEURO-SYMBOLIC EMPIRICAL & ML UTILITY BENCHMARK SUITE")
    print(f"Total Datasets to Evaluate: {len(ALL_DATASETS)}")
    print("=" * 85)

    results = []
    engine = CleaningEngine()

    for ds in ALL_DATASETS:
        name = ds["name"]
        print(f"\n---> Evaluating [{ds['category']}] {name}...")
        rel_p = ds["path"]
        abs_p = os.path.join(WORKSPACE_ROOT, rel_p)

        if not os.path.exists(abs_p):
            print(f"  [SKIPPED] File not found: {abs_p}")
            continue

        file_size_kb = os.path.getsize(abs_p) / 1024.0

        # 1. Ingestion
        t0 = time.time()
        if ds.get("is_excel"):
            df = pd.read_excel(abs_p, dtype=str, engine="openpyxl", nrows=ds.get("max_rows")).fillna("").astype(str)
        else:
            hdr = ds.get("header", 0)
            names = ds.get("names")
            max_r = ds.get("max_rows")
            df = load_table(abs_p, nrows=max_r)
            if names and len(df.columns) == len(names):
                df.columns = names

        t_load = time.time() - t0
        n_rows, n_cols = df.shape

        # 2. Profiling
        t0 = time.time()
        prof = profile_table(df)
        t_prof = time.time() - t0

        # 3. Proposal Generation (with safe imputation enabled)
        t0 = time.time()
        props = engine.generate_proposals(df, prof, enable_imputation=True)
        t_props = time.time() - t0

        # Counts by tier and kind
        tier_counts = {"AUTO": 0, "REVIEW": 0, "FLAG": 0}
        kind_counts = {}
        for p in props:
            tier_counts[p.tier] = tier_counts.get(p.tier, 0) + 1
            kind_counts[p.kind] = kind_counts.get(p.kind, 0) + 1

        # 4. Cleaning Application (AUTO + REVIEW)
        approved = [p for p in props if p.tier in ["AUTO", "REVIEW"]]
        t0 = time.time()
        df_cleaned, diffs = engine.apply(df, approved)
        t_apply = time.time() - t0

        rec = {
            "name": name,
            "category": ds["category"],
            "domain": ds["domain"],
            "file_size_kb": round(file_size_kb, 1),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_cells": n_rows * n_cols,
            "t_load": round(t_load, 3),
            "t_profile": round(t_prof, 3),
            "t_proposals": round(t_props, 3),
            "t_apply": round(t_apply, 3),
            "t_total": round(t_load + t_prof + t_props + t_apply, 3),
            "total_proposals": len(props),
            "tier_auto": tier_counts.get("AUTO", 0),
            "tier_review": tier_counts.get("REVIEW", 0),
            "tier_flag": tier_counts.get("FLAG", 0),
            "kind_counts": kind_counts,
            "cells_repaired": len(diffs)
        }

        # 5. Academic Ground Truth Metrics (if clean.csv exists)
        if ds.get("has_ground_truth") and ds.get("clean_path"):
            clean_p = os.path.join(WORKSPACE_ROOT, ds["clean_path"])
            if os.path.exists(clean_p):
                df_clean_gt = pd.read_csv(
                    clean_p,
                    nrows=ds.get("max_rows"),
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False
                ).fillna("").astype(str)
                m = calculate_metrics(
                    df, df_clean_gt, df_cleaned,
                    numeric_tolerant=True,
                    exclude_first_col_as_index=ds.get("has_index_col", False)
                )
                rec["gt_precision"] = round(m["precision"], 4)
                rec["gt_recall"] = round(m["recall"], 4)
                rec["gt_f1"] = round(m["f1"], 4)
                rec["gt_harm_rate"] = round(m["harm_rate"], 6)
                print(f"  Ground Truth: Prec={m['precision']*100:.2f}%, Rec={m['recall']*100:.2f}%, F1={m['f1']*100:.2f}%, Harm={m['harm_rate']*100:.4f}%")

        # 6. Downstream ML Utility Evaluation
        target_col = ds.get("target_col")
        if target_col and target_col in df.columns and target_col in df_cleaned.columns:
            ml_res = evaluate_downstream_ml(df, df_cleaned, target_column=target_col, random_seed=42)
            if ml_res.get("status") == "success":
                rec["ml_task"] = ml_res["task_type"]
                rec["ml_target"] = target_col
                rec["ml_model"] = ml_res["model_name"]
                rec["ml_dirty_metrics"] = ml_res["dirty_metrics"]
                rec["ml_cleaned_metrics"] = ml_res["cleaned_metrics"]
                rec["ml_deltas"] = ml_res["deltas"]
                rec["ml_improved"] = ml_res["improved"]

                if ml_res["task_type"] == "Classification":
                    d_acc = ml_res["deltas"].get("delta_accuracy", 0.0)
                    d_f1 = ml_res["deltas"].get("delta_f1", 0.0)
                    pct_f1 = ml_res["deltas"].get("pct_gain_f1", 0.0)
                    print(f"  ML Utility (Classif): Dirty F1={ml_res['dirty_metrics']['f1_score']:.4f} -> Clean F1={ml_res['cleaned_metrics']['f1_score']:.4f} (Lift: {pct_f1:+.2f}%)")
                else:
                    d_r2 = ml_res["deltas"].get("delta_r2", 0.0)
                    print(f"  ML Utility (Regress): Dirty R²={ml_res['dirty_metrics']['r2_score']:.4f} -> Clean R²={ml_res['cleaned_metrics']['r2_score']:.4f} (Delta R²: {d_r2:+.4f})")

        results.append(rec)
        print(f"  Execution Time: {rec['t_total']}s | Repaired Cells: {rec['cells_repaired']}")

    out_p = os.path.join(WORKSPACE_ROOT, "eval", "results", "full_neuro_symbolic_benchmark_run.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 85)
    print(f"BENCHMARK RUN COMPLETE! Results saved to: {out_p}")
    print("=" * 85)

if __name__ == "__main__":
    run_all()
