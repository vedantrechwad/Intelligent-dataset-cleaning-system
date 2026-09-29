"""
cleaner/rules/r4_dates.py
R4 Dates:
Convert only when the format is inferable from the column (some value has a part >12 in one position, or ISO).
Otherwise FLAG as ambiguous (e.g. 03/04/2026).
Tier: AUTO if unambiguous, else FLAG.
"""

import re
from typing import List, Set, Dict, Any, Optional, Tuple
import pandas as pd
from cleaner.rules.base import Proposal, CellChange


DATE_SLASH_DASH_RE = re.compile(r"^(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})$")
ISO_DATE_RE = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$")


def propose_r4_dates(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    touched_cells: Optional[Set[tuple]] = None
) -> List[Proposal]:
    """
    Propose R4 date canonicalization or ambiguity flagging.
    """
    if touched_cells is None:
        touched_cells = set()

    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows == 0:
        return proposals

    for col in df.columns:
        series = df[col]
        non_empty = [(idx, val.strip()) for idx, val in series.items() if val != "" and (idx, col) not in touched_cells]
        if len(non_empty) < 5:
            continue

        # Check date-like values
        parsed_dates = []
        for idx, val in non_empty:
            m = DATE_SLASH_DASH_RE.match(val)
            if m:
                p1, p2, p3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                parsed_dates.append((idx, val, p1, p2, p3))

        # Must have >= 60% date-like values to be a date column
        if len(parsed_dates) / len(non_empty) < 0.60:
            continue

        # Check if all dates are already ISO (YYYY-MM-DD)
        is_already_iso = all(ISO_DATE_RE.match(val) for _, val, _, _, _ in parsed_dates)
        if is_already_iso:
            continue

        # Determine column-level inferability
        has_day_first = False   # p1 > 12 and p2 <= 12 and p3 > 31 (DD/MM/YYYY)
        has_month_first = False # p1 <= 12 and p2 > 12 and p3 > 31 (MM/DD/YYYY)

        for _, val, p1, p2, p3 in parsed_dates:
            # Check 4-digit year at end (p3 >= 1000)
            if p3 >= 1000:
                if p1 > 12 and p2 <= 12:
                    has_day_first = True
                elif p2 > 12 and p1 <= 12:
                    has_month_first = True
            # Check 4-digit year at start (p1 >= 1000)
            elif p1 >= 1000:
                # ISO-like YYYY/MM/DD vs YYYY/DD/MM
                if p2 > 12 and p3 <= 12:
                    has_day_first = True  # Year/Day/Month
                elif p3 > 12 and p2 <= 12:
                    has_month_first = True  # Year/Month/Day (ISO)

        # Ambiguity check
        if (has_day_first and has_month_first) or (not has_day_first and not has_month_first):
            # Format cannot be proven from file evidence -> FLAG (never change)
            sample_ambiguous = [val for _, val, _, _, _ in parsed_dates[:3]]
            proposals.append(Proposal(
                id=f"R4_dates_ambiguous_{col}",
                kind="R4_dates",
                tier="FLAG",
                columns=[col],
                description=f"Flag ambiguous date format in column '{col}' (neither MM/DD nor DD/MM inferable)",
                evidence=(
                    f"{len(parsed_dates)} date values detected, but no component > 12 definitively proves day vs month order. "
                    f"Sample values: [{', '.join(sample_ambiguous)}]. Values flagged, not changed."
                ),
                changes=[]
            ))
            continue

        # Unambiguous: format is inferable
        changes: List[CellChange] = []
        if has_day_first:
            detected_format = "DD/MM/YYYY"
            for idx, val, p1, p2, p3 in parsed_dates:
                year = p3 if p3 >= 1000 else (2000 + p3 if p3 < 50 else 1900 + p3)
                day = p1
                month = p2
                if 1 <= month <= 12 and 1 <= day <= 31:
                    new_val = f"{year:04d}-{month:02d}-{day:02d}"
                    if val != new_val:
                        changes.append(CellChange(row=int(idx), column=col, old_value=val, new_value=new_val))
        else:
            detected_format = "MM/DD/YYYY"
            for idx, val, p1, p2, p3 in parsed_dates:
                year = p3 if p3 >= 1000 else (2000 + p3 if p3 < 50 else 1900 + p3)
                month = p1
                day = p2
                if 1 <= month <= 12 and 1 <= day <= 31:
                    new_val = f"{year:04d}-{month:02d}-{day:02d}"
                    if val != new_val:
                        changes.append(CellChange(row=int(idx), column=col, old_value=val, new_value=new_val))

        if changes:
            count = len(changes)
            proposals.append(Proposal(
                id=f"R4_dates_canonical_{col}",
                kind="R4_dates",
                tier="AUTO",
                columns=[col],
                description=f"Canonicalize {count} date values to ISO format (YYYY-MM-DD) in column '{col}'",
                evidence=f"Column format proven as {detected_format} via values with day component > 12. Canonicalized to ISO.",
                changes=changes
            ))

    return proposals
