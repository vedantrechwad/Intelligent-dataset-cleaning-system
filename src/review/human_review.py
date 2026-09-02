import re
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from src.utils.helpers import load_config

def validate_user_correction_input(
    column_name: str,
    issue_type: str,
    user_input_val: Any
) -> Tuple[bool, Any, str]:
    """
    Strictly validates human-entered correction inputs before committing them.
    Ensures dates are genuine, numeric values conform to domain ranges, etc.
    Returns:
      (is_valid, parsed_value, error_message)
    """
    str_val = str(user_input_val).strip()
    if not str_val:
        return False, None, "Correction value cannot be empty or whitespace."

    # 1. Validation for Dates
    if issue_type == "invalid_date" or "date" in column_name.lower():
        try:
            parsed = pd.to_datetime(str_val, format="mixed")
            # Format cleanly as YYYY-MM-DD
            clean_date_str = parsed.strftime("%Y-%m-%d")
            return True, clean_date_str, ""
        except Exception as e:
            return False, None, f"Invalid date format: '{str_val}'. Please enter a valid date (e.g., YYYY-MM-DD)."

    # 2. Validation for Range / Numeric
    cfg = load_config()
    domain_rules = cfg.get("domain_rules", {})
    
    col_lower = column_name.lower()
    for rule_key, bounds in domain_rules.items():
        if rule_key in col_lower:
            try:
                num = float(str_val)
                min_v = bounds.get("min")
                max_v = bounds.get("max")
                if min_v is not None and num < min_v:
                    return False, None, f"{column_name} cannot be less than {min_v} (entered: {num})."
                if max_v is not None and num > max_v:
                    return False, None, f"{column_name} cannot exceed {max_v} (entered: {num})."
                clean_num = int(num) if num.is_integer() else num
                return True, clean_num, ""
            except ValueError:
                return False, None, f"Expected numeric input for {column_name}, received '{str_val}'."

    # Generic Numeric Check
    if issue_type in ["invalid_range", "outlier", "type_inconsistency"]:
        try:
            num = float(str_val)
            clean_num = int(num) if num.is_integer() else num
            return True, clean_num, ""
        except ValueError:
            pass

    # Generic string accepted
    return True, str_val, ""


class HumanReviewManager:
    """
    Tracks and manages issues requiring human-in-the-loop review.
    """
    def __init__(self):
        self.decisions: Dict[str, Dict[str, Any]] = {}

    def record_decision(
        self,
        issue_id: str,
        action: str, # 'accept', 'reject', 'custom'
        row: int,
        column: str,
        original_value: Any,
        issue_type: str,
        corrected_value: Optional[Any] = None,
        notes: str = ""
    ) -> Tuple[bool, str]:
        """
        Records human decision with strict validation.
        """
        if action == "reject":
            self.decisions[issue_id] = {
                "status": "rejected",
                "row": row,
                "column": column,
                "original_value": original_value,
                "corrected_value": original_value,
                "issue_type": issue_type,
                "notes": notes or "User rejected proposed change"
            }
            return True, "Correction rejected."

        # Validate input for accept/custom
        val_to_check = corrected_value
        is_valid, clean_val, err_msg = validate_user_correction_input(column, issue_type, val_to_check)
        if not is_valid:
            return False, err_msg

        self.decisions[issue_id] = {
            "status": "accepted",
            "row": row,
            "column": column,
            "original_value": original_value,
            "corrected_value": clean_val,
            "issue_type": issue_type,
            "notes": notes or "User confirmed correction"
        }
        return True, "Correction verified and saved."

    def get_all_decisions(self) -> Dict[str, Dict[str, Any]]:
        return self.decisions
