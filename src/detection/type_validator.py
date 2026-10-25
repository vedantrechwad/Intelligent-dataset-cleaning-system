import re
from typing import List, Dict, Any
import pandas as pd
from src.utils.helpers import load_config

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100
}

BOOLEAN_MAPPING = {
    "yes": True, "no": False,
    "true": True, "false": False,
    "t": True, "f": False,
    "1": True, "0": False,
    "y": True, "n": False
}

NULL_TOKENS = {"n/a", "na", "null", "none", "?", "-", "undefined", "missing", ""}

def detect_type_inconsistencies(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    type_issues = []

    for col in df.columns:
        series = df[col]
        valid_series = series.dropna()
        if len(valid_series) == 0:
            continue

        # Check for Boolean normalization candidate
        str_series_lower = valid_series.astype(str).str.strip().str.lower()
        is_predominantly_bool = str_series_lower.isin(BOOLEAN_MAPPING.keys()).mean() > 0.8
        
        # Check for Numeric candidate
        numeric_count = pd.to_numeric(valid_series, errors="coerce").notna().sum()
        is_predominantly_numeric = (numeric_count / len(valid_series)) >= 0.70

        # Pre-compute issue actions on unique values to avoid redundant O(N) regex evaluation
        unique_vals = series.dropna().unique()
        val_issue_map = {}

        for val in unique_vals:
            str_val = str(val).strip()
            if not str_val:
                continue
            lower_val = str_val.lower()

            # 1. Null-token normalization
            if lower_val in NULL_TOKENS:
                val_issue_map[val] = {
                    "issue_type": "type_inconsistency",
                    "detection_method": ["type_validator_null_token"],
                    "detection_confidence": 0.99,
                    "correction_confidence": 0.99,
                    "suggested_action": "convert_to_null",
                    "suggested_value": None,
                    "reason": f"Value '{str_val}' is a known null-token and safely converts to NaN",
                    "is_human_review_required": False
                }
                continue

            # 2. Boolean normalization
            if is_predominantly_bool and not pd.api.types.is_bool_dtype(series):
                if lower_val in BOOLEAN_MAPPING:
                    val_issue_map[val] = {
                        "issue_type": "type_inconsistency",
                        "detection_method": ["type_validator_boolean"],
                        "detection_confidence": 0.99,
                        "correction_confidence": 0.99,
                        "suggested_action": "convert_to_boolean",
                        "suggested_value": BOOLEAN_MAPPING[lower_val],
                        "reason": f"Value '{str_val}' safely normalizes to boolean {BOOLEAN_MAPPING[lower_val]}",
                        "is_human_review_required": False
                    }
                    continue

            # 3. Numeric Formatting Normalization (Only flag if formatting differs or value is non-numeric)
            if is_predominantly_numeric and not pd.api.types.is_numeric_dtype(series):
                clean_str = re.sub(r'[$,€£]', '', str_val)
                if clean_str.count(',') >= 1 and clean_str.count('.') <= 1:
                    clean_str = clean_str.replace(',', '')

                try:
                    num = float(clean_str)
                    clean_val = int(num) if num.is_integer() else num
                    if clean_str != str_val:
                        val_issue_map[val] = {
                            "issue_type": "type_inconsistency",
                            "detection_method": ["type_validator_safe_numeric"],
                            "detection_confidence": 1.0,
                            "correction_confidence": 0.99,
                            "suggested_action": "convert_type",
                            "suggested_value": clean_val,
                            "reason": f"Value '{str_val}' safely converts to numeric {clean_val}",
                            "is_human_review_required": False
                        }
                except ValueError:
                    if lower_val in WORD_TO_NUM:
                        suggested = WORD_TO_NUM[lower_val]
                        val_issue_map[val] = {
                            "issue_type": "type_inconsistency",
                            "detection_method": ["type_validator_word_number"],
                            "detection_confidence": 0.95,
                            "correction_confidence": 0.85,
                            "suggested_action": "convert_word_to_number",
                            "suggested_value": suggested,
                            "reason": f"Text word '{str_val}' represents numeric value {suggested}. Review recommended.",
                            "is_human_review_required": True
                        }
                    else:
                        val_issue_map[val] = {
                            "issue_type": "type_inconsistency",
                            "detection_method": ["type_validator_ambiguous"],
                            "detection_confidence": 1.0,
                            "correction_confidence": 0.0,
                            "suggested_action": "flag_for_human_review",
                            "suggested_value": None,
                            "reason": f"Non-numeric value '{str_val}' found in numeric column '{col}'",
                            "is_human_review_required": True
                        }

        # If any unique values have issues, locate rows
        if val_issue_map:
            bad_keys = set(val_issue_map.keys())
            for row_idx, val in series.items():
                if val in bad_keys:
                    issue_meta = val_issue_map[val].copy()
                    issue_meta["row"] = int(row_idx)
                    issue_meta["column"] = col
                    issue_meta["original_value"] = str(val).strip()
                    type_issues.append(issue_meta)

    return type_issues
