import os
import sys
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine
from eval.run_benchmark import eq

def test_numeric_parseability(df: pd.DataFrame) -> Tuple[int, int, float]:
    """Test how many cells that look numeric can be parsed without error."""
    parseable = 0
    total_non_empty = 0
    for col in df.columns:
        for val in df[col]:
            s = str(val).strip()
            if s and s not in ["", "NA", "N/A", "null", "None", "nan", "NaN"]:
                total_non_empty += 1
                try:
                    float(s)
                    parseable += 1
                except ValueError:
                    pass
    pct = (parseable / total_non_empty * 100.0) if total_non_empty > 0 else 100.0
    return parseable, total_non_empty, pct

def count_whitespace_pollution(df: pd.DataFrame) -> int:
    """Count cells with leading or trailing whitespace."""
    count = 0
    for col in df.columns:
        for val in df[col]:
            s = str(val)
            if len(s) > 0 and (s != s.strip()):
                count += 1
    return count

def count_sentinel_tokens(df: pd.DataFrame) -> int:
    """Count dirty sentinel strings like 'NA', 'N/A', 'null', 'None', '?'."""
    sentinels = {"na", "n/a", "null", "none", "?", "-", "--", "missing", "nan"}
    count = 0
    for col in df.columns:
        for val in df[col]:
            s = str(val).strip().lower()
            if s in sentinels:
                count += 1
    return count

def evaluate_dataset(dataset_name: str) -> Dict[str, Any]:
    dirty_path = os.path.join(WORKSPACE_ROOT, "benchmarks", dataset_name, "dirty.csv")
    clean_path = os.path.join(WORKSPACE_ROOT, "benchmarks", dataset_name, "clean.csv")
    
    if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
        return {}
        
    df_dirty = load_table(dirty_path)
    df_clean = load_table(clean_path)
    
    # Run our engine
    prof = profile_table(df_dirty)
    engine = CleaningEngine()
    proposals = engine.generate_proposals(df_dirty, prof)
    approved = [p for p in proposals if p.tier in ["AUTO", "REVIEW"]]
    df_out, diffs = engine.apply(df_dirty, approved)
    
    n_rows = min(len(df_dirty), len(df_clean), len(df_out))
    n_cols = min(len(df_dirty.columns), len(df_clean.columns), len(df_out.columns))
    total_cells = n_rows * n_cols
    
    fixed = 0
    missed = 0
    wrong = 0
    damaged = 0
    clean_cells = 0
    errors_total = 0
    
    diff_categories = {
        "fixed": [],
        "damaged": [],
        "wrong": [],
        "missed": []
    }
    
    for r in range(n_rows):
        for c in range(n_cols):
            d = df_dirty.iat[r, c]
            cl = df_clean.iat[r, c]
            o = df_out.iat[r, c]
            col_name = df_dirty.columns[c]
            
            is_err = not eq(d, cl)
            if is_err:
                errors_total += 1
                if eq(o, cl):
                    fixed += 1
                    if len(diff_categories["fixed"]) < 5:
                        diff_categories["fixed"].append((r, col_name, d, o, cl))
                elif eq(o, d):
                    missed += 1
                    if len(diff_categories["missed"]) < 5:
                        diff_categories["missed"].append((r, col_name, d, o, cl))
                else:
                    wrong += 1
                    if len(diff_categories["wrong"]) < 5:
                        diff_categories["wrong"].append((r, col_name, d, o, cl))
            else:
                clean_cells += 1
                if not eq(o, cl):
                    damaged += 1
                    if len(diff_categories["damaged"]) < 5:
                        diff_categories["damaged"].append((r, col_name, d, o, cl))
                        
    recall = (fixed / errors_total) if errors_total > 0 else 0.0
    denom_prec = fixed + wrong + damaged
    precision = (fixed / denom_prec) if denom_prec > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    harm_rate = (damaged / clean_cells) if clean_cells > 0 else 0.0
    
    # Exact match % between O and C across ALL cells
    exact_match_oc = sum(1 for r in range(n_rows) for c in range(n_cols) if eq(df_out.iat[r, c], df_clean.iat[r, c]))
    agreement_pct = (exact_match_oc / total_cells) * 100.0
    
    # Data Quality Dimensions
    ws_dirty = count_whitespace_pollution(df_dirty)
    ws_clean = count_whitespace_pollution(df_clean)
    ws_out = count_whitespace_pollution(df_out)
    
    sen_dirty = count_sentinel_tokens(df_dirty)
    sen_clean = count_sentinel_tokens(df_clean)
    sen_out = count_sentinel_tokens(df_out)
    
    return {
        "dataset": dataset_name,
        "rows": n_rows,
        "cols": n_cols,
        "total_cells": total_cells,
        "errors_total": errors_total,
        "clean_cells": clean_cells,
        "fixed": fixed,
        "missed": missed,
        "wrong": wrong,
        "damaged": damaged,
        "recall_pct": round(recall * 100, 2),
        "precision_pct": round(precision * 100, 2),
        "f1_pct": round(f1 * 100, 2),
        "harm_rate_pct": round(harm_rate * 100, 4),
        "agreement_with_provided_clean_pct": round(agreement_pct, 2),
        "whitespace_cells": {
            "dirty": ws_dirty,
            "provided_clean": ws_clean,
            "our_cleaned": ws_out
        },
        "sentinel_cells": {
            "dirty": sen_dirty,
            "provided_clean": sen_clean,
            "our_cleaned": sen_out
        },
        "samples": diff_categories
    }

datasets = ["beers", "hospital", "movies_1", "flights", "rayyan", "telco"]
all_res = []
for ds in datasets:
    res = evaluate_dataset(ds)
    if res:
        all_res.append(res)

print(json.dumps(all_res, indent=2))
