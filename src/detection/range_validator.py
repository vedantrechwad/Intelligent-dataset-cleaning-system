import re
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from src.utils.helpers import load_config

def validate_domain_ranges(df: pd.DataFrame, custom_rules: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Validates domain rules and identifies definitely invalid / impossible values.
    Strictly distinguishes between statistical outliers and impossible values.
    Values where the true value cannot safely be inferred are marked for human review.
    """
    if custom_rules is None:
        cfg = load_config()
        custom_rules = cfg.get("domain_rules", {
            "percentage": {"min": 0.0, "max": 100.0},
            "percent": {"min": 0.0, "max": 100.0},
            "age": {"min": 0, "max": 120},
            "salary": {"min": 0.0}
        })

    invalid_issues = []

    # Sort rule keys by length descending so more specific names (e.g. 'percentage') match before substrings (e.g. 'age')
    sorted_rule_keys = sorted(custom_rules.keys(), key=len, reverse=True)

    for col in df.columns:
        col_lower = str(col).lower()
        series = df[col]
        rule = None

        # Match column name against configured domain rules with word boundary
        for rule_key in sorted_rule_keys:
            # Match if whole word or delimited by underscore
            pattern = rf"(^|_|\b){re.escape(rule_key)}($|_|\b)"
            if re.search(pattern, col_lower) or rule_key == col_lower:
                rule = custom_rules[rule_key]
                break

        if not rule:
            continue

        min_val = rule.get("min")
        max_val = rule.get("max")

        for row_idx, val in series.items():
            if pd.isna(val) or val is None:
                continue

            # Convert to float for evaluation
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                continue

            violation = None
            if min_val is not None and num_val < min_val:
                violation = f"{col} cannot be less than {min_val} (found {num_val})"
            elif max_val is not None and num_val > max_val:
                violation = f"{col} cannot exceed realistic threshold of {max_val} (found {num_val})"

            if violation:
                invalid_issues.append({
                    "row": int(row_idx),
                    "column": col,
                    "original_value": num_val,
                    "issue_type": "invalid_range",
                    "detection_method": ["range_validator"],
                    "detection_confidence": 1.0,
                    "correction_confidence": 0.0, # System cannot safely guess the intended value!
                    "suggested_action": "flag_for_human_review",
                    "suggested_value": None,
                    "reason": violation,
                    "is_human_review_required": True
                })

    return invalid_issues
