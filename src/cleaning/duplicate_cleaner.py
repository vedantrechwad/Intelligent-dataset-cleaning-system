from typing import Tuple, List, Dict, Any
import pandas as pd

def clean_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Removes exact identical rows, keeping the first occurrence.
    Returns cleaned dataframe and log of removed rows.
    """
    cleaned_df = df.copy(deep=True)
    duplicate_mask = cleaned_df.duplicated(keep="first")
    removed_indices = cleaned_df[duplicate_mask].index.tolist()

    logs = []
    for idx in removed_indices:
        logs.append({
            "row": int(idx),
            "column": "__all__",
            "original_value": "Duplicate row",
            "issue_type": "exact_duplicate",
            "action": "removed_duplicate_row",
            "corrected_value": None,
            "confidence": 1.0,
            "reason": "Exact duplicate row removed by automated cleaning engine"
        })

    cleaned_df = cleaned_df.drop(index=removed_indices).reset_index(drop=True)
    return cleaned_df, logs
