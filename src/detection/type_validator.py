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

def detect_type_inconsistencies(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Detects type inconsistencies, distinguishing safely convertible representations
    (e.g., stringified numbers "25" in numeric columns) from ambiguous values (e.g., "Thirty").
    """
    type_issues = []

    for col in df.columns:
        series = df[col]
        # Identify predominantly numeric columns
        valid_series = series.dropna()
        if len(valid_series) == 0:
            continue

        # Count how many can convert to float
        numeric_count = pd.to_numeric(valid_series, errors="coerce").notna().sum()
        ratio = numeric_count / len(valid_series)

        # If 70%+ are numeric, this column is expected to be numeric
        if ratio >= 0.70 and not pd.api.types.is_numeric_dtype(series):
            for row_idx, val in series.items():
                if pd.isna(val) or val is None:
                    continue

                str_val = str(val).strip()
                if not str_val:
                    continue

                # Case A: Stringified number (e.g. "25" or "42.5") -> High confidence auto-correct
                try:
                    num = float(str_val)
                    clean_val = int(num) if num.is_integer() else num
                    type_issues.append({
                        "row": int(row_idx),
                        "column": col,
                        "original_value": str_val,
                        "issue_type": "type_inconsistency",
                        "detection_method": ["type_validator_safe_numeric"],
                        "detection_confidence": 1.0,
                        "correction_confidence": 0.99, # Safe deterministic conversion
                        "suggested_action": "convert_type",
                        "suggested_value": clean_val,
                        "reason": f"Value '{str_val}' is stored as string but safely converts to numeric {clean_val}",
                        "is_human_review_required": False
                    })
                except ValueError:
                    # Case B: Word-based representation (e.g. "Thirty")
                    lower_val = str_val.lower()
                    if lower_val in WORD_TO_NUM:
                        suggested = WORD_TO_NUM[lower_val]
                        type_issues.append({
                            "row": int(row_idx),
                            "column": col,
                            "original_value": str_val,
                            "issue_type": "type_inconsistency",
                            "detection_method": ["type_validator_word_number"],
                            "detection_confidence": 0.95,
                            "correction_confidence": 0.85, # Review recommended
                            "suggested_action": "convert_word_to_number",
                            "suggested_value": suggested,
                            "reason": f"Text word '{str_val}' represents numeric value {suggested}. Review recommended.",
                            "is_human_review_required": True
                        })
                    else:
                        # Case C: Ambiguous non-numeric string in numeric column
                        type_issues.append({
                            "row": int(row_idx),
                            "column": col,
                            "original_value": str_val,
                            "issue_type": "type_inconsistency",
                            "detection_method": ["type_validator_ambiguous"],
                            "detection_confidence": 1.0,
                            "correction_confidence": 0.0,
                            "suggested_action": "flag_for_human_review",
                            "suggested_value": None,
                            "reason": f"Non-numeric value '{str_val}' found in numeric column '{col}'",
                            "is_human_review_required": True
                        })

    return type_issues
