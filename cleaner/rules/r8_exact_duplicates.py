"""
cleaner/rules/r8_exact_duplicates.py
R8 Exact duplicates:
Drop exact duplicate rows (keep first) with counts.
Tier: REVIEW.
"""

from typing import List, Set, Dict, Any, Optional
import pandas as pd
from cleaner.rules.base import Proposal


def propose_r8_exact_duplicates(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    touched_cells: Optional[Set[tuple]] = None
) -> List[Proposal]:
    """
    Propose dropping exact duplicate rows (keep first occurrence).
    """
    n_rows = len(df)
    if n_rows < 2:
        return []

    dup_mask = df.duplicated(keep="first")
    dup_indices = [int(idx) for idx in df[dup_mask].index]

    if not dup_indices:
        return []

    count = len(dup_indices)
    share = count / n_rows

    return [
        Proposal(
            id="R8_exact_duplicates",
            kind="R8_exact_duplicates",
            tier="REVIEW",
            columns=list(df.columns),
            description=f"Drop {count} exact duplicate rows (preserving first occurrence)",
            evidence=f"{count} exact duplicate rows detected in dataset ({share*100:.2f}% of rows).",
            dropped_rows=dup_indices,
            n_cells=count * len(df.columns)
        )
    ]
