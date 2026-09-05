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
        for row_idx, val in series.items():
            is_missing = False
            raw_val = val
            
            # Check native null
            if pd.isna(val) or val is None:
                is_missing = True
            elif isinstance(val, str):
                s = val.strip()
                if not s or s.lower() in missing_lookup:
                    is_missing = True

            if is_missing:
                missing_issues.append({
                    "row": int(row_idx),
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
