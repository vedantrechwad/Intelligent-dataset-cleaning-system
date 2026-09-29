"""
cleaner/profile.py
Per-column profiling, type evidence, missing-token counts, and shape analysis.
"""

import re
from typing import Dict, Any, List
import pandas as pd


def compute_shape_signature(val: str) -> str:
    """
    Compute character shape signature:
    digits -> '#', letters -> 'a', spaces and punctuation preserved.
    e.g. '12.0 oz.' -> '#.# aa.'
    """
    sig = []
    for ch in val:
        if ch.isdigit():
            sig.append("#")
        elif ch.isalpha():
            sig.append("a")
        else:
            sig.append(ch)
    return "".join(sig)


def profile_table(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profile table columns: shape frequency tables, missing token counts, cardinality.
    """
    n_rows, n_cols = df.shape
    col_profiles: Dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        non_empty = [v for v in series if v != ""]
        empty_count = n_rows - len(non_empty)

        # Shapes
        shape_counts: Dict[str, int] = {}
        for v in non_empty[:5000]:  # sample for speed on massive tables
            sig = compute_shape_signature(v)
            shape_counts[sig] = shape_counts.get(sig, 0) + 1

        sorted_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)

        col_profiles[col] = {
            "total_rows": n_rows,
            "non_empty_count": len(non_empty),
            "empty_count": empty_count,
            "distinct_count": series.nunique(),
            "dominant_shapes": sorted_shapes[:5],
            "sample_values": list(series.unique()[:5])
        }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": col_profiles
    }
