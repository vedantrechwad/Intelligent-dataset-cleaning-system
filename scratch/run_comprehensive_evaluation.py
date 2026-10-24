"""
scratch/run_comprehensive_evaluation.py
Comprehensive empirical evaluation across academic benchmarks and real-world test datasets.
Collects:
- Dataset characteristics (rows, cols, cells, size, format, domain)
- Timing breakdown (load, profile, proposal, total)
- Memory usage (RSS MB)
- Cleaning actions (proposals by rule, changes, tiers)
- Research metrics on datasets with ground truth (Errors, Fixed, Damaged, Precision, Recall, F1, Harm Rate)
"""

import os
import sys
import time
import json
import psutil
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine
from eval.compare_published import calculate_metrics

process = psutil.Process()

def get_memory_mb() -> float:
    return process.memory_info().rss / (1024 * 1024)

DATASETS = [
    # Academic Benchmark Suite (with clean.csv ground truth)
    {
        "name": "hospital",
        "category": "Academic Benchmark",
        "domain": "Healthcare Operations",
        "path": "benchmarks/hospital/dirty.csv",
        "clean_path": "benchmarks/hospital/clean.csv",
        "has_ground_truth": True,
        "has_index_col": True,
        "sep": ","
    },
    {
        "name": "beers",
        "category": "Academic Benchmark",
        "domain": "Food & Beverage / Brewing",
        "path": "benchmarks/beers/dirty.csv",
        "clean_path": "benchmarks/beers/clean.csv",
        "has_ground_truth": True,
        "has_index_col": True,
        "sep": ","
    },
    {
        "name": "flights",
        "category": "Academic Benchmark",
        "domain": "Aviation & Logistics",
        "path": "benchmarks/flights/dirty.csv",
        "clean_path": "benchmarks/flights/clean.csv",
        "has_ground_truth": True,
        "has_index_col": False,
        "sep": ","
    },
    {
        "name": "movies_1",
        "category": "Academic Benchmark",
        "domain": "Entertainment & Media",
        "path": "benchmarks/movies_1/dirty.csv",
        "clean_path": "benchmarks/movies_1/clean.csv",
        "has_ground_truth": True,
        "has_index_col": False,
        "sep": ","
    },
    {
        "name": "rayyan",
        "category": "Academic Benchmark",
        "domain": "Biomedical Literature",
        "path": "benchmarks/rayyan/dirty.csv",
        "clean_path": "benchmarks/rayyan/clean.csv",
        "has_ground_truth": True,
        "has_index_col": False,
        "sep": ","
    },
    {
        "name": "telco",
        "category": "Academic Benchmark",
        "domain": "Telecommunications Churn",
        "path": "benchmarks/telco/dirty.csv",
        "clean_path": "benchmarks/telco/clean.csv",
        "has_ground_truth": True,
        "has_index_col": False,
        "sep": ","
    },
    {
        "name": "tax_sample_25k",
        "category": "Academic Benchmark",
        "domain": "IRS Tax Records",
        "path": "benchmarks/tax/dirty.csv",
        "clean_path": "benchmarks/tax/clean.csv",
        "has_ground_truth": True,
        "has_index_col": False,
        "sep": ",",
        "max_rows": 25000
    },

    # New Test Datasets (Famous Real-World & Machine Learning Datasets)
    {
        "name": "gender_submission (Titanic)",
        "category": "New Test Dataset",
        "domain": "Maritime / Passenger Survival",
        "path": "new test datasets/gender_submission.csv",
        "has_ground_truth": False,
        "sep": ","
    },
    {
        "name": "wdbc (Breast Cancer Wisconsin)",
        "category": "New Test Dataset",
        "domain": "Medical Diagnostic Oncology",
        "path": "new test datasets/breast+cancer+wisconsin+diagnostic/wdbc.data",
        "has_ground_truth": False,
        "sep": ",",
        "header": None,
        "names": ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
    },
    {
        "name": "student_math (Student Performance)",
        "category": "New Test Dataset",
        "domain": "Education & Academic Outcomes",
        "path": "new test datasets/student+performance/student-mat.csv",
        "has_ground_truth": False,
        "sep": ";"
    },
    {
        "name": "student_portuguese",
        "category": "New Test Dataset",
        "domain": "Education & Academic Outcomes",
        "path": "new test datasets/student+performance/student-por.csv",
        "has_ground_truth": False,
        "sep": ";"
    },
    {
        "name": "bank_marketing (Sample)",
        "category": "New Test Dataset",
        "domain": "Banking & Direct Marketing",
        "path": "new test datasets/bank+marketing/bank.csv",
        "has_ground_truth": False,
        "sep": ";"
    },
    {
        "name": "bank_marketing (Full 45k)",
        "category": "New Test Dataset",
        "domain": "Banking & Direct Marketing",
        "path": "new test datasets/bank+marketing/bank-full.csv",
        "has_ground_truth": False,
        "sep": ";"
    },
    {
        "name": "bank_additional (Full 41k)",
        "category": "New Test Dataset",
        "domain": "Banking & Macroeconomic Indicators",
        "path": "new test datasets/bank+marketing/bank-additional/bank-additional-full.csv",
        "has_ground_truth": False,
        "sep": ";"
    },
    {
        "name": "adult_census_income (32.5k)",
        "category": "New Test Dataset",
        "domain": "Socioeconomic & Census Classification",
        "path": "new test datasets/adult (1)/adult.data",
        "has_ground_truth": False,
        "sep": ",",
        "header": None,
        "names": [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]
    },
    {
        "name": "credit_card_fraud (284k)",
        "category": "New Test Dataset",
        "domain": "Financial Fraud & Anomaly Detection",
        "path": "new test datasets/creditcard.csv/creditcard.csv",
        "has_ground_truth": False,
        "sep": ","
    },
    {
        "name": "online_retail_sample_50k",
        "category": "New Test Dataset",
        "domain": "E-Commerce & Transaction Analytics",
        "path": "new test datasets/online+retail/Online Retail.xlsx",
        "has_ground_truth": False,
        "is_excel": True,
        "max_rows": 50000
    }
]

def run_benchmarks():
    results = []
    print("=" * 80)
    print("STARTING COMPREHENSIVE EMPIRICAL EVALUATION")
    print("=" * 80)

    for ds in DATASETS:
        name = ds["name"]
        print(f"\nProcessing [{ds['category']}] {name}...")
        
        path = os.path.join(WORKSPACE_ROOT, ds["path"])
        if not os.path.exists(path):
            print(f"  Skipping {name}: file not found at {path}")
            continue

        file_size_bytes = os.path.getsize(path)
        mem_start = get_memory_mb()

        # Ingestion
        t0 = time.time()
        if ds.get("is_excel"):
            max_r = ds.get("max_rows")
            df = pd.read_excel(path, dtype=str, engine="openpyxl", nrows=max_r)
            df = df.fillna("").astype(str)
        else:
            sep = ds.get("sep", ",")
            header = ds.get("header", 0)
            names = ds.get("names")
            max_r = ds.get("max_rows")
            df = pd.read_csv(
                path,
                sep=sep,
                header=header,
                names=names,
                nrows=max_r,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                encoding="utf-8",
                low_memory=False
            )
            df = df.fillna("").astype(str)

        t_load = time.time() - t0
        n_rows, n_cols = df.shape
        n_cells = n_rows * n_cols

        # Profiling
        t0 = time.time()
        prof = profile_table(df)
        t_prof = time.time() - t0

        # Proposal Generation
        t0 = time.time()
        eng = CleaningEngine()
        props = eng.generate_proposals(df, prof)
        t_props = time.time() - t0

        mem_end = get_memory_mb()
        peak_mem_delta = max(0.0, mem_end - mem_start)

        # Count proposals by rule and tier
        tier_counts = {"AUTO": 0, "REVIEW": 0, "FLAG": 0}
        kind_counts = {}
        total_cell_changes = 0

        for p in props:
            tier_counts[p.tier] = tier_counts.get(p.tier, 0) + 1
            kind_counts[p.kind] = kind_counts.get(p.kind, 0) + 1
            total_cell_changes += len(p.changes)

        # Cleaning Application
        approved_auto = [p for p in props if p.tier == "AUTO"]
        approved_all = [p for p in props if p.tier in ["AUTO", "REVIEW"]]

        t0 = time.time()
        df_cleaned_auto, diffs_auto = eng.apply(df, approved_auto)
        df_cleaned_all, diffs_all = eng.apply(df, approved_all)
        t_apply = time.time() - t0

        rec = {
            "name": name,
            "category": ds["category"],
            "domain": ds["domain"],
            "file_size_kb": file_size_bytes / 1024,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_cells": n_cells,
            "t_load": t_load,
            "t_profile": t_prof,
            "t_proposals": t_props,
            "t_apply": t_apply,
            "t_total": t_load + t_prof + t_props + t_apply,
            "peak_mem_mb": mem_end,
            "mem_delta_mb": peak_mem_delta,
            "proposals_total": len(props),
            "tier_auto": tier_counts.get("AUTO", 0),
            "tier_review": tier_counts.get("REVIEW", 0),
            "tier_flag": tier_counts.get("FLAG", 0),
            "kind_counts": kind_counts,
            "cell_changes_proposed": total_cell_changes,
            "cells_changed_auto": len(diffs_auto),
            "cells_changed_all": len(diffs_all)
        }

        # If ground truth exists, compute precision, recall, F1, harm rate
        if ds.get("has_ground_truth") and ds.get("clean_path"):
            clean_p = os.path.join(WORKSPACE_ROOT, ds["clean_path"])
            if os.path.exists(clean_p):
                df_clean = pd.read_csv(
                    clean_p,
                    nrows=ds.get("max_rows"),
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                    encoding="utf-8",
                    low_memory=False
                ).fillna("").astype(str)

                has_idx = ds.get("has_index_col", False)
                
                # Metrics for AUTO proposals only
                m_auto = calculate_metrics(df, df_clean, df_cleaned_auto, numeric_tolerant=True, exclude_first_col_as_index=has_idx)
                # Metrics for ALL (AUTO + REVIEW)
                m_all = calculate_metrics(df, df_clean, df_cleaned_all, numeric_tolerant=True, exclude_first_col_as_index=has_idx)

                rec["errors_true"] = m_all["E_count"]
                rec["fixed_auto"] = m_auto["fixed"]
                rec["damaged_auto"] = m_auto["damaged"]
                rec["precision_auto"] = m_auto["precision"]
                rec["recall_auto"] = m_auto["recall"]
                rec["f1_auto"] = m_auto["f1"]
                rec["harm_rate_auto"] = m_auto["harm_rate"]

                rec["fixed_all"] = m_all["fixed"]
                rec["damaged_all"] = m_all["damaged"]
                rec["precision_all"] = m_all["precision"]
                rec["recall_all"] = m_all["recall"]
                rec["f1_all"] = m_all["f1"]
                rec["harm_rate_all"] = m_all["harm_rate"]

        results.append(rec)
        print(f"  Done in {rec['t_total']:.2f}s | Props: {rec['proposals_total']} (Auto: {rec['tier_auto']}, Rev: {rec['tier_review']}, Flag: {rec['tier_flag']})")
        if "f1_all" in rec:
            print(f"  Benchmark Metrics: Prec={rec['precision_all']*100:.2f}%, Rec={rec['recall_all']*100:.2f}%, F1={rec['f1_all']*100:.2f}%, Harm={rec['harm_rate_all']*100:.3f}%")

    out_file = os.path.join(WORKSPACE_ROOT, "eval", "results", "comprehensive_benchmark_run.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_file}")

if __name__ == "__main__":
    run_benchmarks()
