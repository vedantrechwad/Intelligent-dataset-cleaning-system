from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

def handle_outliers(
    df: pd.DataFrame,
    outlier_issues: List[Dict[str, Any]],
    strategy: str = "flag_only"
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Handles numerical outliers based on user policy:
      - 'flag_only': No modification, records audit acknowledgement (Recommended default).
      - 'cap': Winsorizes values to IQR bounds.
      - 'remove': Drops rows containing extreme outliers.
    """
    cleaned_df = df.copy(deep=True)
    logs = []

    if strategy == "flag_only" or not outlier_issues:
        for issue in outlier_issues:
            logs.append({
                "row": issue.get("row"),
                "column": issue.get("column"),
                "original_value": issue.get("original_value"),
                "issue_type": "outlier",
                "action": "DETECT_ONLY",
                "method": "flag_only",
                "corrected_value": issue.get("original_value"),
                "detection_confidence": issue.get("detection_confidence", 0.8),
                "correction_confidence": 0.0,
                "reason": "Outlier retained per 'flag_only' policy"
            })
        return cleaned_df, logs

    elif strategy == "cap":
        # Group issues by column
        col_issues = {}
        for issue in outlier_issues:
            col = issue.get("column")
            if col in cleaned_df.columns:
                col_issues.setdefault(col, []).append(issue)

        for col, issues in col_issues.items():
            valid_vals = pd.to_numeric(cleaned_df[col], errors="coerce").dropna()
            if len(valid_vals) < 5:
                continue

            q1 = float(np.percentile(valid_vals, 25))
            q3 = float(np.percentile(valid_vals, 75))
            iqr = q3 - q1
            lower_bound = round(q1 - 1.5 * iqr, 3)
            upper_bound = round(q3 + 1.5 * iqr, 3)

            for issue in issues:
                row_idx = issue.get("row")
                orig_val = issue.get("original_value")
                if row_idx in cleaned_df.index:
                    capped_val = upper_bound if orig_val > upper_bound else lower_bound
                    cleaned_df.at[row_idx, col] = capped_val
                    logs.append({
                        "row": int(row_idx),
                        "column": col,
                        "original_value": orig_val,
                        "issue_type": "outlier",
                        "action": "CORRECTION",
                        "method": "cap_winsorize",
                        "corrected_value": capped_val,
                        "detection_confidence": 1.0,
                        "correction_confidence": 0.85,
                        "reason": f"Value {orig_val} capped to IQR bound ({capped_val})"
                    })

    elif strategy == "remove":
        rows_to_remove = set()
        for issue in outlier_issues:
            row_idx = issue.get("row")
            if row_idx in cleaned_df.index:
                rows_to_remove.add(row_idx)
                logs.append({
                    "row": int(row_idx),
                    "column": issue.get("column"),
                    "original_value": issue.get("original_value"),
                    "issue_type": "outlier",
                    "action": "CORRECTION",
                    "method": "remove_row",
                    "corrected_value": None,
                    "detection_confidence": 1.0,
                    "correction_confidence": 0.85,
                    "reason": "Row removed due to extreme outlier per user policy"
                })

        cleaned_df = cleaned_df.drop(index=list(rows_to_remove)).reset_index(drop=True)

    return cleaned_df, logs
