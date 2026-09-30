from typing import List, Dict, Any
import pandas as pd
import numpy as np
from src.utils.helpers import load_config

def detect_missing_values(df: pd.DataFrame, custom_missing: List[str] = None) -> List[Dict[str, Any]]:
    """
    Detects missing values including NaN, None, empty strings, whitespace,
    and configured representations (e.g. 'N/A', 'null', 'unknown').
    """
    if custom_missing is None:
        cfg = load_config()
        custom_missing = cfg.get("missing_representations", [
            "", " ", "nan", "NaN", "none", "None", "null", "NULL",
            "n/a", "N/A", "na", "NA", "?", "-"
        ])

    missing_issues = []
    # Set of lowercase representations for fast matching
    missing_lookup = {str(m).strip().lower() for m in custom_missing}

    for col in df.columns:
        series = df[col]
        # Check native nulls
        null_mask = series.isna()
        
        # Check string representations using vectorized string operations
        # Convert only non-nulls to string for comparison to avoid 'nan' string matches unless specified
        str_series = series.astype(str).str.strip().str.lower()
        str_mask = str_series.isin(missing_lookup)
        empty_str_mask = str_series == ""
        
        missing_mask = null_mask | str_mask | empty_str_mask
        
        if missing_mask.any():
            missing_indices = missing_mask[missing_mask].index
            # Cap at 5000 issues per column to prevent memory explosions on massive datasets
            for idx in missing_indices[:5000]:
                raw_val = series.loc[idx]
                missing_issues.append({
                    "row": int(idx),
                    "column": col,
                    "original_value": None if pd.isna(raw_val) else str(raw_val),
                    "issue_type": "missing_value",
                    "detection_method": ["missing_detector"],
                    "detection_confidence": 1.0,
                    "correction_confidence": 0.0, # Requires imputation strategy
                    "suggested_action": "impute",
                    "suggested_value": None,
                    "reason": f"Value is missing or matches configured missing token '{raw_val}'",
                    "is_human_review_required": False # Handled by chosen imputation policy
                })

    return missing_issues
