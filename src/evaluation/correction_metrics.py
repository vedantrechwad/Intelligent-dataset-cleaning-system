from typing import List, Dict, Any
import pandas as pd
import numpy as np

def evaluate_correction_performance(
    ground_truth: List[Dict[str, Any]],
    cleaned_df: pd.DataFrame,
    audit_trail: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes scientific correction accuracy:
      Accuracy = Number Correctly Restored / Number of Automatic Corrections Attempted
    Separately distinguishes automated AI accuracy from human-assisted corrections and unresolved issues.
    """
    # Map ground truth: (row, column) -> original_value
    gt_map = {(gt["row"], gt["column"]): gt for gt in ground_truth if gt.get("column") != "__all__"}

    # Identify automatic actions vs user actions from audit trail
    auto_attempts = 0
    auto_correct_restored = 0
    user_assisted_count = 0

    for act in audit_trail:
        action_type = act.get("action", "")
        row = act.get("row")
        col = act.get("column")
        cell_key = (row, col)

        if action_type == "user_corrected":
            user_assisted_count += 1
            continue

        if cell_key in gt_map:
            auto_attempts += 1
            expected_orig = gt_map[cell_key].get("original_value")

            # Check what's now in cleaned_df
            if row in cleaned_df.index and col in cleaned_df.columns:
                actual_cleaned = cleaned_df.at[row, col]

                # Check if restored closely or exactly
                is_restored = False
                if str(actual_cleaned).strip().lower() == str(expected_orig).strip().lower():
                    is_restored = True
                elif isinstance(actual_cleaned, (int, float)) and isinstance(expected_orig, (int, float)):
                    if abs(float(actual_cleaned) - float(expected_orig)) < 1e-2:
                        is_restored = True

                if is_restored:
                    auto_correct_restored += 1

    auto_accuracy = (auto_correct_restored / auto_attempts) if auto_attempts > 0 else 0.0

    # Unresolved ground truth issues
    unresolved_count = 0
    for cell_key, gt in gt_map.items():
        row, col = cell_key
        expected_orig = gt.get("original_value")
        if row in cleaned_df.index and col in cleaned_df.columns:
            actual_val = cleaned_df.at[row, col]
            if pd.isna(actual_val) or str(actual_val).strip().lower() != str(expected_orig).strip().lower():
                unresolved_count += 1

    return {
        "automatic_corrections_attempted": auto_attempts,
        "automatic_corrections_restored": auto_correct_restored,
        "automatic_correction_accuracy": round(auto_accuracy * 100.0, 2),
        "user_assisted_corrections": user_assisted_count,
        "unresolved_injected_issues": unresolved_count,
        "total_ground_truth_anomalies": len(ground_truth)
    }
