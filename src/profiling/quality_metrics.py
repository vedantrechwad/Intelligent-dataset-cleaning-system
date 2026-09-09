from typing import Dict, Any, List
import pandas as pd
from src.utils.helpers import load_config, logger

def compute_quality_score(
    df: pd.DataFrame,
    detected_issues: List[Dict[str, Any]] = None,
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Computes a reproducible, explainable Data Quality Score (0-100)
    across five standard data engineering quality dimensions:
      1. Completeness: penalty for missing cells
      2. Uniqueness: penalty for duplicate records
      3. Validity: penalty for invalid dates, domain range violations, format errors
      4. Consistency: penalty for casing, whitespace, and category typos
      5. Anomaly Quality: penalty for extreme statistical outliers
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

    # 1. Completeness Score
    missing_cells = int(df.isna().sum().sum())
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

    # Apply a geometric penalty scaling (e.g., multiplier of 10 for severity)
    # This ensures tabular datasets drop visibly in quality score for structural/formatting errors.
    penalty_multiplier = 10.0

    # 3. Validity Score
    validity_penalty = min(1.0, (invalid_count / total_cells) * penalty_multiplier)
    validity_score = max(0.0, 100.0 * (1.0 - validity_penalty))

    # 4. Consistency Score
    consistency_penalty = min(1.0, (inconsistent_count / total_cells) * penalty_multiplier)
    consistency_score = max(0.0, 100.0 * (1.0 - consistency_penalty))

    # 5. Anomaly Quality Score
    numeric_cells = sum([
        len(df[col].dropna()) for col in df.columns 
        if pd.api.types.is_numeric_dtype(df[col])
    ])
    numeric_cells = max(1, numeric_cells)
    anomaly_score = max(0.0, min(100.0, 100.0 * (1.0 - (outlier_count / numeric_cells))))

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
        explanation.append(f"Completeness is {round(completeness_score, 1)}% due to {missing_cells} missing cell(s).")
    if uniqueness_score < 95:
        explanation.append(f"Uniqueness is {round(uniqueness_score, 1)}% due to {duplicate_rows} duplicate row(s).")
    if validity_score < 95:
        explanation.append(f"Validity is {round(validity_score, 1)}% due to {invalid_count} invalid value(s).")
    if consistency_score < 95:
        explanation.append(f"Consistency is {round(consistency_score, 1)}% due to {inconsistent_count} category/text variation(s).")
    if anomaly_score < 95:
        explanation.append(f"Anomaly quality is {round(anomaly_score, 1)}% due to {outlier_count} statistical outlier(s).")
        
    if not explanation:
        explanation.append("Excellent data hygiene across all dimensions.")

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
