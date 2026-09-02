import re
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

DATE_KEYWORDS = ["date", "dob", "birth", "joined", "timestamp", "created_at", "updated_at", "time"]

def is_date_column_candidate(col_name: str, series: pd.Series) -> bool:
    """Determines if a column is intended to represent dates."""
    col_lower = str(col_name).lower()
    if any(k in col_lower for k in DATE_KEYWORDS):
        return True
    
    # Check if samples look like dates
    sample = series.dropna().astype(str).head(10)
    if sample.empty:
        return False
        
    date_pattern = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$")
    matching = sample.apply(lambda x: bool(date_pattern.match(x.strip()))).sum()
    return matching >= 3


def validate_dates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects invalid dates, invalid months, invalid days (e.g. 2025-02-30, 22/13/2020),
    and unparseable date strings.
    Invalid dates are strictly flagged for human review. The system never invents a date.
    """
    invalid_date_issues = []

    for col in df.columns:
        series = df[col]
        if not is_date_column_candidate(col, series):
            continue

        for row_idx, val in series.items():
            if pd.isna(val) or val is None:
                continue

            str_val = str(val).strip()
            if not str_val:
                continue

            # Attempt parsing
            is_valid = False
            reason = ""

            # Check explicit known invalid patterns like month > 12 or day > 31
            match_slash = re.match(r"^(\d{1,4})[/-](\d{1,4})[/-](\d{1,4})$", str_val)
            if match_slash:
                p1, p2, p3 = int(match_slash.group(1)), int(match_slash.group(2)), int(match_slash.group(3))
                # Check for month 13, etc.
                if p1 > 12 and p2 > 12:
                    is_valid = False
                    reason = f"Invalid month/day: both '{p1}' and '{p2}' exceed 12"
                elif (p1 == 2 and p2 > 29) or (p2 == 2 and p1 > 29):
                    is_valid = False
                    reason = f"Invalid day for February in date '{str_val}'"
                elif p2 == 0 or p1 == 0:
                    is_valid = False
                    reason = f"Month or day cannot be zero in '{str_val}'"

            if not reason:
                try:
                    # Parse with pandas
                    pd.to_datetime(str_val, format="mixed", errors="raise")
                    is_valid = True
                except Exception as e:
                    is_valid = False
                    reason = f"Unparseable date: '{str_val}'. Reason: {str(e)}"

            if not is_valid:
                invalid_date_issues.append({
                    "row": int(row_idx),
                    "column": col,
                    "original_value": str_val,
                    "issue_type": "invalid_date",
                    "detection_method": ["date_validator"],
                    "detection_confidence": 1.0,
                    "correction_confidence": 0.0, # Cannot invent a date safely
                    "suggested_action": "flag_for_human_review",
                    "suggested_value": None,
                    "reason": reason or f"Date '{str_val}' is logically invalid",
                    "is_human_review_required": True
                })

    return invalid_date_issues
