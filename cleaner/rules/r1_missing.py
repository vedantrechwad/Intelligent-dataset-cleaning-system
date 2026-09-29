"""
cleaner/rules/r1_missing.py
R1 Missing tokens rule:
A candidate token (na, n/a, null, none, nan, ?, --, -, undefined, #n/a, nil, or empty-after-trim)
becomes empty only if it is rare in that column (<=5% of values).
If it is a frequent value (e.g. NA as a legitimate code, None as a real category) propose REVIEW instead.
"""

import re
from typing import List, Set, Dict, Any, Optional
import pandas as pd
from cleaner.rules.base import Proposal, CellChange


CANDIDATE_MISSING_TOKENS: Set[str] = {
    "na", "n/a", "null", "none", "nan", "?", "--", "-", "undefined",
    "#n/a", "nil", "n.a.", "n/d", "#na", "n/k"
}


def propose_r1_missing(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    touched_cells: Optional[Set[tuple]] = None
) -> List[Proposal]:
    """
    Propose R1 missing token cleanups.
    """
    if touched_cells is None:
        touched_cells = set()

    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows == 0:
        return proposals

    for col in df.columns:
        series = df[col]
        # 1. Check whitespace-only cells (empty after trim)
        empty_trim_changes: List[CellChange] = []
        for r_idx, val in series.items():
            if (r_idx, col) in touched_cells:
                continue
            if val != "" and val.strip() == "":
                empty_trim_changes.append(
                    CellChange(row=int(r_idx), column=col, old_value=val, new_value="")
                )

        if empty_trim_changes:
            count = len(empty_trim_changes)
            proposals.append(Proposal(
                id=f"R1_missing_trim_{col}",
                kind="R1_missing_tokens",
                tier="AUTO",
                columns=[col],
                description=f"Convert {count} whitespace-only blank cells to empty string in column '{col}'",
                evidence=f"{count} cells in column '{col}' contain only whitespace ({count/n_rows*100:.1f}% of rows).",
                changes=empty_trim_changes
            ))

        # 2. Check candidate missing tokens
        token_groups: Dict[str, List[int]] = {}
        original_token_repr: Dict[str, str] = {}

        for r_idx, val in series.items():
            if (r_idx, col) in touched_cells:
                continue
            stripped = val.strip()
            low = stripped.lower()
            if low in CANDIDATE_MISSING_TOKENS:
                if low not in token_groups:
                    token_groups[low] = []
                    original_token_repr[low] = stripped
                token_groups[low].append(int(r_idx))

        for tok_key, row_indices in token_groups.items():
            count = len(row_indices)
            share = count / n_rows
            sample_repr = original_token_repr[tok_key]

            changes = [
                CellChange(
                    row=r,
                    column=col,
                    old_value=series.iat[r],
                    new_value=""
                )
                for r in row_indices
            ]

            clean_tok_name = re.sub(r"[^a-zA-Z0-9_]", "_", sample_repr)
            prop_id = f"R1_missing_{col}_{clean_tok_name}"

            if share <= 0.05:
                # Rare in column -> AUTO
                proposals.append(Proposal(
                    id=prop_id,
                    kind="R1_missing_tokens",
                    tier="AUTO",
                    columns=[col],
                    description=f"Convert rare disguised missing token '{sample_repr}' to empty string in column '{col}'",
                    evidence=f"Token '{sample_repr}' appears in {count} of {n_rows} rows ({share*100:.1f}%), within <= 5.0% rarity threshold.",
                    changes=changes
                ))
            else:
                # Frequent in column -> REVIEW
                proposals.append(Proposal(
                    id=prop_id,
                    kind="R1_missing_tokens",
                    tier="REVIEW",
                    columns=[col],
                    description=f"Review candidate missing token '{sample_repr}' in column '{col}' (frequent value: possible valid domain code/category)",
                    evidence=f"Token '{sample_repr}' appears in {count} of {n_rows} rows ({share*100:.1f}%), exceeding 5.0% rarity threshold. Propose verification before blanking.",
                    changes=changes
                ))

    return proposals
