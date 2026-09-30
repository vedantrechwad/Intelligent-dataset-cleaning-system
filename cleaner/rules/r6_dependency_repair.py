"""
cleaner/rules/r6_dependency_repair.py
R6 Dependency repair:
Discover approximate functional dependencies A -> B over column pairs:
group rows by A (groups of size >=2), confidence = sum of majority counts / sum of group sizes.
Propose when confidence >= 0.95, A is not a unique key, and >=30% of rows are in groups of size >=2.
Repair violations to the group majority only when the majority has support >=2 and a >=2/3 share;
also fill blanks in B from the group majority.
Tier: REVIEW.
"""

from typing import List, Set, Dict, Any, Optional, Tuple
import pandas as pd
from cleaner.rules.base import Proposal, CellChange


def propose_r6_dependency_repair(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    touched_cells: Optional[Set[tuple]] = None,
    min_confidence: float = 0.95,
    min_grouped_row_share: float = 0.30,
    max_columns: int = 40,
    sample_threshold: int = 25_000
) -> List[Proposal]:
    """
    Propose R6 functional dependency repairs.
    """
    if touched_cells is None:
        touched_cells = set()

    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows < 5:
        return proposals

    work_df = df.sample(n=sample_threshold, random_state=42) if n_rows > sample_threshold else df
    cols = list(work_df.columns)[:max_columns]

    # Find valid functional dependencies
    valid_deps = []

    for a in cols:
        a_series = work_df[a]
        non_blank_a = a_series[a_series != ""]
        if len(non_blank_a) < 5:
            continue

        # Skip near-unique key
        if non_blank_a.nunique() / len(non_blank_a) > 0.90:
            continue

        vc_a = non_blank_a.value_counts()
        groups_ge_2 = vc_a[vc_a >= 2]
        if groups_ge_2.empty:
            continue
        grouped_rows = groups_ge_2.sum()
        if (grouped_rows / len(non_blank_a)) < min_grouped_row_share:
            continue

        for b in cols:
            if a == b:
                continue

            b_series = work_df[b]
            non_blank_b = b_series[b_series != ""]
            if len(non_blank_b) < 5:
                continue
            overall_b_dominant_share = non_blank_b.value_counts().iloc[0] / len(non_blank_b)
            # Skip if B is practically constant table-wide
            if overall_b_dominant_share >= 0.85:
                continue

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
                if conf >= min_confidence and conf > (overall_b_dominant_share + 0.02):
                    valid_deps.append({
                        "col_a": a,
                        "col_b": b,
                        "confidence": conf,
                        "majority_count": sum(maj_counts),
                        "total_count": sum(grp_sizes)
                    })

    # Sort dependencies by confidence and majority support descending
    valid_deps.sort(key=lambda d: (d["confidence"], d["majority_count"]), reverse=True)

    local_touched = set(touched_cells)

    for dep in valid_deps:
        col_a = dep["col_a"]
        col_b = dep["col_b"]
        conf = dep["confidence"]

        changes: List[CellChange] = []
        n_repairs = 0
        n_fills = 0

        # Fast lookup mapping: group non_blanks by col_a to get majority value
        sub_all = df[[col_a, col_b]][df[col_a] != ""]
        non_blanks_b = sub_all[sub_all[col_b] != ""]
        if len(non_blanks_b) < 2:
            continue

        a_to_maj: Dict[str, str] = {}
        for a_val, grp_df in non_blanks_b.groupby(col_a):
            if len(grp_df) >= 2:
                vc_b = grp_df[col_b].value_counts()
                maj_val = vc_b.index[0]
                maj_count = vc_b.iloc[0]
                if maj_count >= 2 and (maj_count / len(grp_df)) >= (2.0 / 3.0):
                    a_to_maj[a_val] = maj_val

        if not a_to_maj:
            continue

        # Fast iterrows pass
        for idx, row in sub_all.iterrows():
            idx = int(idx)
            if (idx, col_b) in local_touched:
                continue
            a_val = row[col_a]
            if a_val in a_to_maj:
                maj_val = a_to_maj[a_val]
                cur_b = row[col_b]
                if cur_b != "" and cur_b != maj_val:
                    changes.append(
                        CellChange(row=idx, column=col_b, old_value=cur_b, new_value=maj_val)
                    )
                    n_repairs += 1
                    local_touched.add((idx, col_b))
                elif cur_b == "":
                    changes.append(
                        CellChange(row=idx, column=col_b, old_value="", new_value=maj_val)
                    )
                    n_fills += 1
                    local_touched.add((idx, col_b))

        if changes:
            proposals.append(Proposal(
                id=f"R6_dependency_{col_a}_{col_b}",
                kind="R6_dependency_repair",
                tier="REVIEW",
                columns=[col_a, col_b],
                description=f"Repair '{col_b}' violations and fill blanks using functional dependency '{col_a}' -> '{col_b}'",
                evidence=(
                    f"Approximate functional dependency '{col_a}' -> '{col_b}' observed with {conf*100:.1f}% confidence. "
                    f"Repaired {n_repairs} violation cells and filled {n_fills} blank cells to group majority (support >= 2, share >= 66.7%)."
                ),
                changes=changes
            ))

    return proposals
