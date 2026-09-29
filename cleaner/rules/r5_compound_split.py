"""
cleaner/rules/r5_compound_split.py
R5 Compound fields:
If values in column A frequently end with a token that is a frequent value of another column B
(learned from B's observed values), propose splitting; fill B only if blank or equal.
Tier: REVIEW.
"""

from typing import List, Set, Dict, Any, Optional, Tuple
import pandas as pd
from cleaner.rules.base import Proposal, CellChange


def propose_r5_compound_split(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    touched_cells: Optional[Set[tuple]] = None
) -> List[Proposal]:
    """
    Propose R5 compound field splits.
    """
    if touched_cells is None:
        touched_cells = set()

    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows < 5:
        return proposals

    cols = list(df.columns)

    for col_b in cols:
        b_vals = [v.strip() for v in df[col_b] if v != ""]
        if len(b_vals) < 5:
            continue

        # Learn frequent observed values of B
        b_vc = pd.Series(b_vals).value_counts()
        frequent_b = set(b_vc[(b_vc >= 3) & ((b_vc / len(b_vals)) >= 0.005)].index)
        if not frequent_b:
            continue

        for col_a in cols:
            if col_a == col_b:
                continue

            a_non_empty = [v for v in df[col_a] if v != ""]
            if len(a_non_empty) < 5:
                continue

            matches_a: List[Tuple[int, str, str]] = []  # (row, old_a, new_a)
            matches_b: List[Tuple[int, str, str]] = []  # (row, old_b, new_b)

            for idx, val in df[col_a].items():
                r_idx = int(idx)
                if (r_idx, col_a) in touched_cells or (r_idx, col_b) in touched_cells:
                    continue

                if val == "":
                    continue

                parts = val.strip().rsplit(" ", 1)
                if len(parts) == 2:
                    prefix, suffix = parts[0].strip(), parts[1].strip()
                    if suffix in frequent_b and len(prefix) > 0:
                        cur_b = df.at[r_idx, col_b].strip()
                        # "fill B only if blank or equal"
                        if cur_b == "" or cur_b == suffix:
                            matches_a.append((r_idx, val, prefix))
                            if cur_b == "":
                                matches_b.append((r_idx, df.at[r_idx, col_b], suffix))

            # Threshold: must be frequent in column A (>= 5 rows and >= 1% of non-blank A),
            # AND must actually fill blanks in column B (>= 3 blanks filled) to avoid stripping common vocabulary words from names
            if len(matches_a) >= 5 and (len(matches_a) / len(a_non_empty)) >= 0.01 and len(matches_b) >= 3:
                changes = [
                    CellChange(row=r, column=col_a, old_value=old_v, new_value=new_v)
                    for r, old_v, new_v in matches_a
                ]
                changes.extend([
                    CellChange(row=r, column=col_b, old_value=old_v, new_value=new_v)
                    for r, old_v, new_v in matches_b
                ])

                count_a = len(matches_a)
                count_b = len(matches_b)
                proposals.append(Proposal(
                    id=f"R5_compound_split_{col_a}_{col_b}",
                    kind="R5_compound_split",
                    tier="REVIEW",
                    columns=[col_a, col_b],
                    description=f"Split compound values in column '{col_a}' and populate column '{col_b}'",
                    evidence=(
                        f"{count_a} values in '{col_a}' frequently end with tokens from '{col_b}'. "
                        f"Propose splitting suffix from '{col_a}' and filling {count_b} blank cells in '{col_b}'."
                    ),
                    changes=changes
                ))

    return proposals
