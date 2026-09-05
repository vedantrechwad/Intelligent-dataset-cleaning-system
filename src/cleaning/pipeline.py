from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
from src.cleaning.duplicate_cleaner import clean_exact_duplicates
from src.cleaning.category_cleaner import clean_categorical_inconsistencies
from src.cleaning.type_cleaner import clean_data_types
from src.cleaning.missing_cleaner import clean_missing_values
from src.cleaning.outlier_handler import handle_outliers
from src.utils.helpers import load_config, logger

def execute_cleaning_pipeline(
    df: pd.DataFrame,
    detected_issues: List[Dict[str, Any]],
    user_corrections: Optional[Dict[str, Any]] = None,
    cleaning_options: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Executes the deterministic end-to-end cleaning pipeline.
    Combines high-confidence automated corrections with user-approved human review corrections.
    Returns:
      (cleaned_dataframe, audit_log_records)
    """
    cfg = load_config()
    defaults = cfg.get("cleaning_defaults", {})
    opts = cleaning_options or {}

    remove_dups = opts.get("remove_exact_duplicates", defaults.get("remove_exact_duplicates", True))
    num_missing_strat = opts.get("missing_numeric_strategy", defaults.get("missing_numeric_strategy", "median"))
    cat_missing_strat = opts.get("missing_categorical_strategy", defaults.get("missing_categorical_strategy", "mode"))
    outlier_strat = opts.get("outlier_strategy", defaults.get("outlier_strategy", "flag_only"))
    impute_missing = opts.get("impute_missing", True)

    cleaned_df = df.copy(deep=True)
    audit_trail = []

    # Step 1: Apply User-Provided Review Corrections First
    if user_corrections:
        for issue_id, action_info in user_corrections.items():
            status = action_info.get("status")
            corrected_val = action_info.get("corrected_value")
            row_idx = action_info.get("row")
            col = action_info.get("column")
            orig_val = action_info.get("original_value")

            if status == "accepted" and row_idx in cleaned_df.index and col in cleaned_df.columns:
                cleaned_df.at[row_idx, col] = corrected_val
                audit_trail.append({
                    "issue_id": issue_id,
                    "row": int(row_idx),
                    "column": col,
                    "original_value": orig_val,
                    "issue_type": action_info.get("issue_type", "user_reviewed"),
                    "action": "HUMAN_REVIEW",
                    "status": "accepted",
                    "corrected_value": corrected_val,
                    "detection_confidence": 1.0,
                    "correction_confidence": 1.0,
                    "reason": f"Applied human review correction: {action_info.get('notes', 'User verified')}"
                })
            elif status == "rejected":
                audit_trail.append({
                    "issue_id": issue_id,
                    "row": int(row_idx) if row_idx is not None else None,
                    "column": col,
                    "original_value": orig_val,
                    "issue_type": action_info.get("issue_type", "user_reviewed"),
                    "action": "HUMAN_REVIEW",
                    "status": "rejected",
                    "corrected_value": orig_val,
                    "detection_confidence": 1.0,
                    "correction_confidence": 1.0,
                    "reason": "User rejected proposed modification"
                })

    # Step 2: High-confidence typo corrections
    auto_typos = [
        issue for issue in detected_issues
        if issue.get("routing_decision") == "AUTO_CORRECT" and issue.get("issue_type") == "spelling_typo"
    ]
    for issue in auto_typos:
        row_idx = issue.get("row")
        col = issue.get("column")
        target_val = issue.get("suggested_value")
        if row_idx in cleaned_df.index and col in cleaned_df.columns and target_val:
            cleaned_df.at[row_idx, col] = target_val
            audit_trail.append({
                "issue_id": issue.get("issue_id"),
                "row": int(row_idx),
                "column": col,
                "original_value": issue.get("original_value"),
                "issue_type": "spelling_typo",
                "action": "CORRECTION",
                "method": "rapidfuzz",
                "corrected_value": target_val,
                "detection_confidence": issue.get("detection_confidence", 0.96),
                "correction_confidence": issue.get("correction_confidence", 0.96),
                "reason": issue.get("reason", "High-confidence typo correction")
            })

    # Step 3: Categorical Whitespace and Casing Normalization
    cleaned_df, cat_logs = clean_categorical_inconsistencies(cleaned_df)
    audit_trail.extend(cat_logs)

    # Step 4: Safe Type Conversion
    cleaned_df, type_logs = clean_data_types(cleaned_df)
    audit_trail.extend(type_logs)

    # Step 5: Exact Duplicates Removal
    if remove_dups:
        cleaned_df, dup_logs = clean_exact_duplicates(cleaned_df)
        audit_trail.extend(dup_logs)

    # Step 6: Missing Value Imputation
    if impute_missing:
        cleaned_df, miss_logs = clean_missing_values(
            cleaned_df,
            numeric_strategy=num_missing_strat,
            categorical_strategy=cat_missing_strat
        )
        audit_trail.extend(miss_logs)

    # Step 7: Outlier Handling
    outlier_issues = [i for i in detected_issues if i.get("issue_type") == "outlier"]
    cleaned_df, outlier_logs = handle_outliers(cleaned_df, outlier_issues, strategy=outlier_strat)
    audit_trail.extend(outlier_logs)

    logger.info(f"Cleaning pipeline completed. Rows: {len(cleaned_df)}, modifications logged: {len(audit_trail)}")
    return cleaned_df, audit_trail
