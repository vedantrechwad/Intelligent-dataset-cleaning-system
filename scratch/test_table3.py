import os
import sys
import pandas as pd
import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.corruption_engine import DataCorruptionEngine
from src.detection.anomaly_detector import detect_all_issues
from src.evaluation.detection_metrics import evaluate_detection_performance

def run_benchmark(n_runs=10):
    clean_csv = os.path.join("sample_data", "clean_benchmark_dataset.csv")
    clean_df = pd.read_csv(clean_csv)
    
    # We want to measure the 6 categories from Table 3:
    # 1. Missing values
    # 2. Categorical typos
    # 3. Exact/near duplicates
    # 4. Statistical outliers
    # 5. Range violations
    # 6. Invalid dates
    # And Overall
    
    # Store aggregate TP, FP, FN across runs
    categories = [
        "Missing values",
        "Categorical typos",
        "Exact/near duplicates",
        "Statistical outliers",
        "Range violations",
        "Invalid dates"
    ]
    
    agg = {cat: {"tp": 0, "fp": 0, "fn": 0, "gt": 0} for cat in categories}
    overall_agg = {"tp": 0, "fp": 0, "fn": 0, "gt": 0}
    
    # Type mapping from corruption_type / issue_type to Table 3 categories
    def map_to_table_category(t):
        t = str(t).lower()
        if "missing" in t or "null" in t:
            return "Missing values"
        elif "typo" in t or "casing" in t or "spelling" in t or "whitespace" in t or "inconsistency" in t:
            return "Categorical typos"
        elif "duplicate" in t:
            return "Exact/near duplicates"
        elif "outlier" in t:
            return "Statistical outliers"
        elif "range" in t or "domain" in t:
            return "Range violations"
        elif "date" in t:
            return "Invalid dates"
        return None

    for seed in range(42, 42 + n_runs):
        corrupter = DataCorruptionEngine(seed=seed)
        corrupted_df, ground_truth = corrupter.inject_corruptions(
            clean_df,
            missing_rate=0.08,
            duplicate_count=4,
            outlier_count=5,
            invalid_numeric_count=4,
            invalid_date_count=4,
            spelling_typo_count=5,
            casing_typo_count=5
        )
        
        det_issues = detect_all_issues(corrupted_df)
        
        # Ground truth mapping: key -> category
        # Row-wide vs cell-level
        gt_cells = {}
        for gt in ground_truth:
            cat = map_to_table_category(gt.get("corruption_type"))
            if not cat:
                continue
            r = gt.get("row")
            c = gt.get("column")
            gt_cells[(r, c)] = cat
            agg[cat]["gt"] += 1
            overall_agg["gt"] += 1
            
        # Detected mapping: key -> category
        det_cells = {}
        for issue in det_issues:
            cat = map_to_table_category(issue.get("issue_type"))
            if not cat:
                continue
            r = issue.get("row")
            c = issue.get("column")
            det_cells[(r, c)] = cat
            
        # Calculate TP, FN, FP per category
        for (r, c), cat in gt_cells.items():
            if (r, c) in det_cells:
                # Detected
                agg[cat]["tp"] += 1
                overall_agg["tp"] += 1
            else:
                agg[cat]["fn"] += 1
                overall_agg["fn"] += 1
                
        for (r, c), cat in det_cells.items():
            if (r, c) not in gt_cells:
                # False positive
                agg[cat]["fp"] += 1
                overall_agg["fp"] += 1

    # Print Table 3
    print("\nTable 3: Detection Performance by Issue Type")
    print("-" * 65)
    print(f"{'Issue Type':<25} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    print("-" * 65)
    
    for cat in categories:
        tp = agg[cat]["tp"]
        fp = agg[cat]["fp"]
        fn = agg[cat]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{cat:<25} {prec:<12.3f} {rec:<12.3f} {f1:<10.3f}")
        
    print("-" * 65)
    tot_tp = overall_agg["tp"]
    tot_fp = overall_agg["fp"]
    tot_fn = overall_agg["fn"]
    tot_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    tot_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    tot_f1 = (2 * tot_prec * tot_rec) / (tot_prec + tot_rec) if (tot_prec + tot_rec) > 0 else 0.0
    print(f"{'Overall':<25} {tot_prec:<12.3f} {tot_rec:<12.3f} {tot_f1:<10.3f}")
    print("-" * 65)

if __name__ == "__main__":
    run_benchmark(10)
