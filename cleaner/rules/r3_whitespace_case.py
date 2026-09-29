"""
cleaner/rules/r3_whitespace_case.py
R3 Whitespace and case:
Trim leading and trailing whitespace; collapse repeated spaces where evidenced by column data;
case-variant groups map to the most frequent form only if variants are identical ignoring case/whitespace
and column is categorical (not code-like or free-text).
Tier: AUTO.
"""

import re
from typing import List, Set, Dict, Any, Optional, Tuple
import pandas as pd
from cleaner.rules.base import Proposal, CellChange


def is_code_or_free_text_column(series: pd.Series) -> Tuple[bool, bool]:
    """
    Determine if column is:
    - is_code: short codes (median length <= 3 or mostly short abbreviations)
    - is_free_text: unique free text / structured records (high cardinality or long text)
    """
    non_empty = [v for v in series if v != ""]
    if not non_empty:
        return True, False

    lengths = [len(v) for v in non_empty]
    sorted_lens = sorted(lengths)
    median_len = sorted_lens[len(sorted_lens) // 2]
    n_unique = series.nunique()
    total_non_empty = len(non_empty)

    is_code = median_len <= 3 or all(len(v.strip()) <= 4 for v in non_empty[:50])
    is_free_text = (median_len > 40) or (n_unique > 100 and (n_unique / total_non_empty) > 0.20)
    
    # Check for structured records (e.g. JSON, bibtex)
    sample_starts = [v.strip()[:2] for v in non_empty[:20]]
    if any(st in ['{"', '[\"', "{'", "['"] for st in sample_starts):
        is_free_text = True

    return is_code, is_free_text


def propose_r3_whitespace_case(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    touched_cells: Optional[Set[tuple]] = None
) -> List[Proposal]:
    """
    Propose R3 whitespace and casing cleanups.
    """
    if touched_cells is None:
        touched_cells = set()

    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows == 0:
        return proposals

    for col in df.columns:
        series = df[col]
        is_code, is_free_text = is_code_or_free_text_column(series)
        existing_trimmed_values = set(v.strip() for v in series if v != "")

        # 1. Identify case-variant groups only for non-code, non-free-text columns
        dominant_form_map: Dict[str, str] = {}
        if not is_code and not is_free_text:
            groups: Dict[str, Dict[str, int]] = {}
            for r_idx, val in series.items():
                if (r_idx, col) in touched_cells:
                    continue
                trimmed = val.strip()
                if trimmed == "":
                    continue
                key = trimmed.lower()
                if key not in groups:
                    groups[key] = {}
                groups[key][trimmed] = groups[key].get(trimmed, 0) + 1

            for key, variant_counts in groups.items():
                if len(variant_counts) > 1:
                    total_in_cluster = sum(variant_counts.values())
                    dominant, dom_count = max(variant_counts.items(), key=lambda item: (item[1], item[0]))
                    # Dominant form must have strong evidence (>= 70% of cluster and at least 3 occurrences)
                    if dom_count >= 3 and (dom_count / total_in_cluster) >= 0.70:
                        dominant_form_map[key] = dominant

        # 2. Collect changes
        whitespace_changes: List[CellChange] = []
        casing_changes: List[CellChange] = []

        for r_idx, val in series.items():
            if (r_idx, col) in touched_cells:
                continue
            if val == "":
                continue

            # First: Trim leading/trailing whitespace
            v_trimmed = val.strip()
            if v_trimmed == "":
                # Whitespace-only blank handled by R1 empty-after-trim
                continue

            # Second: Collapse internal repeated whitespace only if single-spaced version exists in the data
            v_collapsed = re.sub(r"\s+", " ", v_trimmed)
            if v_collapsed != v_trimmed and v_collapsed in existing_trimmed_values:
                base_candidate = v_collapsed
            else:
                base_candidate = v_trimmed

            # Third: Apply dominant casing if eligible
            target_val = dominant_form_map.get(base_candidate.lower(), base_candidate)

            if val == target_val:
                continue

            if val.strip() == target_val:
                whitespace_changes.append(
                    CellChange(row=int(r_idx), column=col, old_value=val, new_value=target_val)
                )
            else:
                casing_changes.append(
                    CellChange(row=int(r_idx), column=col, old_value=val, new_value=target_val)
                )

        if whitespace_changes:
            count = len(whitespace_changes)
            proposals.append(Proposal(
                id=f"R3_whitespace_{col}",
                kind="R3_whitespace_case",
                tier="AUTO",
                columns=[col],
                description=f"Trim leading/trailing and normalize whitespace in {count} cells in column '{col}'",
                evidence=f"{count} cells ({count/n_rows*100:.1f}% of column) contain irregular leading/trailing or repeated whitespace.",
                changes=whitespace_changes
            ))

        if casing_changes:
            count = len(casing_changes)
            sample_examples = [f"'{c.old_value}' -> '{c.new_value}'" for c in casing_changes[:3]]
            proposals.append(Proposal(
                id=f"R3_case_{col}",
                kind="R3_whitespace_case",
                tier="AUTO",
                columns=[col],
                description=f"Normalize casing variants to dominant form in {count} cells in column '{col}'",
                evidence=(
                    f"{count} cells in column '{col}' match dominant categorical casing variants. "
                    f"Sample mappings: [{', '.join(sample_examples)}]."
                ),
                changes=casing_changes
            ))

    return proposals
