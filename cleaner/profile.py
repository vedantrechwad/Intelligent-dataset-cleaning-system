"""
cleaner/profile.py
Per-column shape profile, type evidence, missing-token evidence, and functional dependencies.
Motto: Fix what can be proven, flag what can't.
"""

import re
from typing import Dict, Any, List, Set, Tuple, Optional
import pandas as pd
from cleaner.rules.r1_missing import CANDIDATE_MISSING_TOKENS
from cleaner.rules.r2_numeric import parse_numeric_cell


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


def discover_candidate_dependencies(
    df: pd.DataFrame,
    min_confidence: float = 0.95,
    min_grouped_row_share: float = 0.30,
    max_columns: int = 40,
    sample_threshold: int = 200_000
) -> List[Dict[str, Any]]:
    """
    Discover approximate functional dependencies A -> B over column pairs.
    Bounded search: skip near-unique columns, <=40 columns, sample groups on >200k rows.
    Confidence = sum(majority counts) / sum(group sizes).
    """
    n_rows = len(df)
    if n_rows < 5:
        return []

    # Sample rows if > 200k
    work_df = df.sample(n=sample_threshold, random_state=42) if n_rows > sample_threshold else df
    cols = list(work_df.columns)[:max_columns]

    candidate_deps = []

    for a in cols:
        a_series = work_df[a]
        n_unique_a = a_series.nunique()
        # Skip near-unique columns or columns with all unique keys
        if n_unique_a / len(work_df) > 0.90:
            continue

        vc_a = a_series.value_counts()
        groups_ge_2 = vc_a[vc_a >= 2]
        if groups_ge_2.empty:
            continue
        grouped_rows = groups_ge_2.sum()
        if (grouped_rows / len(work_df)) < min_grouped_row_share:
            continue

        for b in cols:
            if a == b:
                continue

            # Sub-dataframe with non-blanks
            sub = work_df[[a, b]][(work_df[a] != "") & (work_df[b] != "")]
            if len(sub) < 5:
                continue

            maj_counts = []
            grp_sizes = []

            for _, grp_df in sub.groupby(a):
                if len(grp_df) >= 2:
                    vc_b = grp_df[b].value_counts()
                    maj_counts.append(vc_b.iloc[0])
                    grp_sizes.append(len(grp_df))

            if grp_sizes and sum(grp_sizes) > 0:
                conf = sum(maj_counts) / sum(grp_sizes)
                if conf >= min_confidence:
                    candidate_deps.append({
                        "col_a": a,
                        "col_b": b,
                        "confidence": conf,
                        "majority_count": sum(maj_counts),
                        "total_count": sum(grp_sizes)
                    })

    return candidate_deps


def profile_table(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produce rich per-column profiles:
    - shape frequency table and minority shapes (< 5%)
    - missing-token counts
    - numeric-parse rate after normalization
    - cardinality (n_unique, uniqueness ratio)
    - candidate functional dependencies A -> B
    Complexity: O(rows x columns).
    """
    n_rows, n_cols = df.shape
    col_profiles: Dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        non_empty = [v for v in series if v != ""]
        empty_count = n_rows - len(non_empty)
        total_non_empty = len(non_empty)

        # 1. Shapes and Minority Shapes
        shape_counts: Dict[str, int] = {}
        for v in non_empty[:5000]:  # sample for speed on massive tables
            sig = compute_shape_signature(v)
            shape_counts[sig] = shape_counts.get(sig, 0) + 1

        sample_total = min(total_non_empty, 5000)
        sorted_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_shapes = sorted_shapes[:5]
        minority_shapes = [
            (sig, cnt) for sig, cnt in sorted_shapes
            if sample_total > 0 and (cnt / sample_total) < 0.05
        ]

        # 2. Missing Token Evidence
        missing_token_counts: Dict[str, int] = {}
        for v in series:
            s_low = v.strip().lower()
            if s_low in CANDIDATE_MISSING_TOKENS or (v != "" and v.strip() == ""):
                tok_name = "whitespace_only" if v.strip() == "" else v.strip()
                missing_token_counts[tok_name] = missing_token_counts.get(tok_name, 0) + 1

        # 3. Numeric Parse Rate after Normalization
        numeric_parseable = 0
        for v in non_empty:
            if parse_numeric_cell(v) is not None:
                numeric_parseable += 1

        num_parse_rate = (numeric_parseable / total_non_empty) if total_non_empty > 0 else 0.0

        # 4. Cardinality
        distinct_count = series.nunique()
        unique_ratio = (distinct_count / n_rows) if n_rows > 0 else 0.0

        col_profiles[col] = {
            "total_rows": n_rows,
            "non_empty_count": total_non_empty,
            "empty_count": empty_count,
            "distinct_count": distinct_count,
            "unique_ratio": unique_ratio,
            "dominant_shapes": dominant_shapes,
            "minority_shapes": minority_shapes,
            "missing_token_counts": missing_token_counts,
            "numeric_parse_rate": num_parse_rate,
            "sample_values": list(series.unique()[:5])
        }

    # 5. Candidate Functional Dependencies
    candidate_deps = discover_candidate_dependencies(df)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": col_profiles,
        "candidate_dependencies": candidate_deps
    }
