from typing import Dict, Any, List, Optional
import pandas as pd
from src.utils.helpers import load_config, logger

SENTINEL_NULLS = {
    "", "nan", "none", "null", "n/a", "na", "?", "<na>", "#n/a", "--", "-", 
    "undefined", "unknown", "error", "missing", "blank", "<null>", "nil", 
    "n.a.", "n/d", "#na", "n/k"
}

def compute_defect_resolution_rate(
    initial_defects: int,
    repaired_defects: int
) -> Dict[str, Any]:
    """
    Computes the Defect Resolution Rate (DRR) percentage:
      DRR = (repaired_defects / initial_defects) * 100
    Provides an honest, defect-focused metric of cleaning efficiency
    independent of total table cell count dilution.
    """
    initial = max(0, int(initial_defects))
    repaired = max(0, int(repaired_defects))
    
    if initial <= 0:
        rate = 100.0
    else:
        rate = round(min(100.0, (repaired / initial) * 100.0), 1)
        
    return {
        "initial_defects": initial,
        "repaired_defects": repaired,
        "remaining_defects": max(0, initial - repaired),
        "resolution_rate_pct": rate
    }

def compute_quality_score(
    df: pd.DataFrame,
    detected_issues: List[Dict[str, Any]] = None,
    weights: Dict[str, float] = None,
    initial_issue_count: Optional[int] = None,
    defects_repaired: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes a reproducible, explainable Data Quality Score (0-100)
    across five standard data engineering quality dimensions:
      1. Completeness: proportion of non-null, non-sentinel cells
      2. Uniqueness: proportion of non-duplicate records
      3. Validity: smooth hyperbolic penalty for invalid formats, ranges, and types
      4. Consistency: smooth hyperbolic penalty for casing, whitespace, and category typos
      5. Anomaly Quality: smooth penalty for extreme statistical outliers
    """
    if weights is None:
        cfg = load_config()
        weights = cfg.get("quality_score_weights", {
            "completeness": 0.25,
            "consistency": 0.20,
            "validity": 0.25,
            "uniqueness": 0.15,
            "anomaly_quality": 0.15
        })

    row_count = len(df)
    col_count = len(df.columns)
    total_cells = max(1, row_count * col_count)

    # 1. Completeness Score (Harmonized across raw string tables and typed DataFrames)
    is_na_mask = df.isna()
    str_na_mask = df.astype(str).apply(lambda col: col.str.strip().str.lower().isin(SENTINEL_NULLS))
    missing_cells = int((is_na_mask | str_na_mask).sum().sum())
    completeness_score = max(0.0, min(100.0, 100.0 * (1.0 - (missing_cells / total_cells))))

    # 2. Uniqueness Score
    duplicate_rows = int(df.duplicated().sum())
    uniqueness_score = max(0.0, min(100.0, 100.0 * (1.0 - (duplicate_rows / max(1, row_count)))))

    # Issue counts by dimension from detected issues list
    invalid_count = 0
    inconsistent_count = 0
    outlier_count = 0
    
    if detected_issues:
        for issue in detected_issues:
            itype = issue.get("issue_type", "")
            if itype in ["invalid_date", "invalid_range", "invalid_type", "domain_violation", "invalid_format"]:
                invalid_count += 1
            elif itype in ["outlier", "extreme_outlier"]:
                outlier_count += 1
            elif itype == "missing_value":
                pass  # Handled by Completeness dimension natively
            elif itype == "exact_duplicate":
                pass  # Handled by Uniqueness dimension natively
            elif itype == "type_inconsistency" and issue.get("suggested_action") in ["convert_type", "convert_to_boolean", "convert_to_null"]:
                pass  # Safe schema typecasts should not heavily penalize cell-level consistency
            else:
                # Catch-all for formatting, near_duplicates, typos, casing, whitespace
                inconsistent_count += 1

    # Smooth continuous hyperbolic penalty curve: score = 100 / (1 + gamma * error_rate)
    # Strictly monotonic (every defect repaired gives positive score gain) and never clips prematurely to 0
    gamma = 5.0

    # 3. Validity Score
    r_val = invalid_count / total_cells
    validity_score = max(0.0, min(100.0, 100.0 / (1.0 + gamma * r_val)))

    # 4. Consistency Score
    r_cons = inconsistent_count / total_cells
    consistency_score = max(0.0, min(100.0, 100.0 / (1.0 + gamma * r_cons)))

    # 5. Anomaly Quality Score
    numeric_cells = sum([
        len(df[col].dropna()) for col in df.columns 
        if pd.api.types.is_numeric_dtype(df[col])
    ])
    numeric_cells = max(1, numeric_cells)
    r_anom = outlier_count / numeric_cells
    anomaly_score = max(0.0, min(100.0, 100.0 / (1.0 + gamma * r_anom)))

    # Weighted Overall Score
    w_comp = weights.get("completeness", 0.25)
    w_cons = weights.get("consistency", 0.20)
    w_val = weights.get("validity", 0.25)
    w_uniq = weights.get("uniqueness", 0.15)
    w_anom = weights.get("anomaly_quality", 0.15)
    
    total_weight = w_comp + w_cons + w_val + w_uniq + w_anom
    if total_weight <= 0:
        total_weight = 1.0

    overall_score = (
        (w_comp * completeness_score) +
        (w_cons * consistency_score) +
        (w_val * validity_score) +
        (w_uniq * uniqueness_score) +
        (w_anom * anomaly_score)
    ) / total_weight

    overall_score = round(overall_score, 2)

    explanation = []
    if completeness_score < 95:
        explanation.append(f"Completeness is {round(completeness_score, 1)}% due to {missing_cells:,} missing cell(s).")
    if uniqueness_score < 95:
        explanation.append(f"Uniqueness is {round(uniqueness_score, 1)}% due to {duplicate_rows:,} duplicate row(s).")
    if validity_score < 95:
        explanation.append(f"Validity is {round(validity_score, 1)}% due to {invalid_count:,} invalid value(s).")
    if consistency_score < 95:
        explanation.append(f"Consistency is {round(consistency_score, 1)}% due to {inconsistent_count:,} category/text variation(s).")
    if anomaly_score < 95:
        explanation.append(f"Anomaly quality is {round(anomaly_score, 1)}% due to {outlier_count:,} statistical outlier(s).")
        
    if not explanation:
        explanation.append("Excellent data hygiene across all dimensions.")

    # Defect Resolution Rate calculation if comparative context is supplied
    drr_metric = None
    if initial_issue_count is not None:
        rep = defects_repaired if defects_repaired is not None else max(0, initial_issue_count - (len(detected_issues) if detected_issues else 0))
        drr_metric = compute_defect_resolution_rate(initial_issue_count, rep)

    return {
        "overall_score": overall_score,
        "grade": "A" if overall_score >= 90 else ("B" if overall_score >= 80 else ("C" if overall_score >= 70 else "D")),
        "dimensions": {
            "completeness": round(completeness_score, 2),
            "uniqueness": round(uniqueness_score, 2),
            "validity": round(validity_score, 2),
            "consistency": round(consistency_score, 2),
            "anomaly_quality": round(anomaly_score, 2)
        },
        "weights": weights,
        "defect_resolution_rate": drr_metric,
        "metrics_summary": {
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "invalid_count": invalid_count,
            "inconsistent_count": inconsistent_count,
            "outlier_count": outlier_count,
            "total_cells": total_cells
        },
        "explanation": " ".join(explanation)
    }

