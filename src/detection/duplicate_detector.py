from typing import List, Dict, Any
import pandas as pd
from rapidfuzz import fuzz
from src.utils.helpers import load_config, logger

def detect_exact_duplicates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects completely identical rows across all columns.
    Identifies duplicate row index and corresponding first-seen row index.
    """
    duplicate_issues = []
    if df.empty:
        return duplicate_issues

    seen_signatures = {}
    for idx in range(len(df)):
        # Convert row values to a hashable tuple representation
        row_tuple = tuple(
            None if pd.isna(v) else (str(v) if isinstance(v, (str, bytes)) else v)
            for v in df.iloc[idx].values
        )

        if row_tuple in seen_signatures:
            original_idx = seen_signatures[row_tuple]
            duplicate_issues.append({
                "row": int(idx),
                "column": "__all__",
                "original_value": f"Duplicate of row {original_idx}",
                "issue_type": "exact_duplicate",
                "detection_method": ["exact_duplicate_detector"],
                "detection_confidence": 1.0,
                "correction_confidence": 1.0,
                "suggested_action": "remove_duplicate",
                "suggested_value": None,
                "reason": f"Row {idx} is completely identical to row {original_idx}",
                "is_human_review_required": False
            })
        else:
            seen_signatures[row_tuple] = idx

    return duplicate_issues


def detect_near_duplicates(
    df: pd.DataFrame, 
    similarity_threshold: float = None, 
    max_sample_rows: int = 1000
) -> List[Dict[str, Any]]:
    """
    Detects potential near-duplicate rows using RapidFuzz string similarity.
    Near duplicates are flagged for user review and are never automatically removed.
    """
    cfg = load_config()
    if similarity_threshold is None:
        similarity_threshold = cfg.get("thresholds", {}).get("near_duplicate_similarity_threshold", 90.0)

    near_duplicates = []
    if len(df) < 2:
        return near_duplicates

    # Exclude exact duplicates first
    exact_dup_mask = df.duplicated(keep=False)
    non_exact_df = df[~exact_dup_mask]
    if len(non_exact_df) < 2:
        # Also check entire df if non_exact is empty
        non_exact_df = df.drop_duplicates(keep="first")
        if len(non_exact_df) < 2:
            return near_duplicates

    sample_df = non_exact_df.head(max_sample_rows)
    row_signatures = sample_df.apply(
        lambda r: " | ".join(str(v).strip().lower() for v in r.values if pd.notna(v) and str(v).strip()),
        axis=1
    )

    indices = list(row_signatures.index)
    signatures = list(row_signatures.values)
    n = len(indices)

    for i in range(n):
        sig_i = signatures[i]
        if not sig_i or len(sig_i) < 5:
            continue
        for j in range(i + 1, min(i + 50, n)):
            sig_j = signatures[j]
            if not sig_j or len(sig_j) < 5:
                continue
            
            sim = fuzz.token_sort_ratio(sig_i, sig_j)
            if sim >= similarity_threshold and sim < 100.0:
                near_duplicates.append({
                    "row": int(indices[j]),
                    "column": "__all__",
                    "original_value": f"Similar to row {indices[i]} (Similarity: {sim:.1f}%)",
                    "issue_type": "near_duplicate",
                    "detection_method": ["rapidfuzz_near_duplicate"],
                    "detection_confidence": round(sim / 100.0, 3),
                    "correction_confidence": 0.80,
                    "suggested_action": "flag_for_review",
                    "suggested_value": f"Row {indices[i]}",
                    "reason": f"Row {indices[j]} has {sim:.1f}% similarity to row {indices[i]}.",
                    "is_human_review_required": True
                })

    return near_duplicates
