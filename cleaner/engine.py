"""
cleaner/engine.py
CleaningEngine enforces core safety invariants:
1. A cell changes ONLY if an approved rule covers it.
2. Every change logs rule_id, old_value, new_value.
3. Untouched cells stay byte-identical.
4. One cell is changed by at most one proposal.
5. Fixed application order: R1 (missing) > R2 (numeric) > R3 (whitespace/case) > ...
6. Deterministic and idempotent: apply twice == apply once.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
from cleaner.rules.base import Proposal, CellChange, RULE_ORDER_MAP
from cleaner.rules.r1_missing import propose_r1_missing
from cleaner.rules.r2_numeric import propose_r2_numeric
from cleaner.rules.r3_whitespace_case import propose_r3_whitespace_case


class CleaningEngine:
    def __init__(self):
        pass

    def generate_proposals(
        self,
        df: pd.DataFrame,
        profile: Optional[Dict[str, Any]] = None
    ) -> List[Proposal]:
        """
        Generate reviewable proposals in strict fixed application order.
        Guarantees:
        - One cell is changed by at most one proposal.
        """
        all_proposals: List[Proposal] = []
        touched_cells: Set[Tuple[int, str]] = set()

        # Step 1: R1 Missing Tokens
        r1_props = propose_r1_missing(df, profile=profile, touched_cells=touched_cells)
        for p in r1_props:
            for c in p.changes:
                touched_cells.add((c.row, c.column))
            all_proposals.append(p)

        # Step 2: R2 Numeric Canonicalization
        r2_props = propose_r2_numeric(df, profile=profile, touched_cells=touched_cells)
        for p in r2_props:
            for c in p.changes:
                touched_cells.add((c.row, c.column))
            all_proposals.append(p)

        # Step 3: R3 Whitespace and Case Normalization
        r3_props = propose_r3_whitespace_case(df, profile=profile, touched_cells=touched_cells)
        for p in r3_props:
            for c in p.changes:
                touched_cells.add((c.row, c.column))
            all_proposals.append(p)

        return all_proposals

    def apply(
        self,
        df: pd.DataFrame,
        approved_proposals: List[Proposal]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Apply approved proposals to the DataFrame in strict order rank.
        
        Returns:
            (cleaned_df, diff_records)
        """
        # Sort proposals by fixed application order
        sorted_props = sorted(approved_proposals, key=lambda p: p.order_rank)

        cleaned_df = df.copy(deep=True)
        diff_records: List[Dict[str, Any]] = []
        modified_cells: Set[Tuple[int, str]] = set()

        for prop in sorted_props:
            # FLAG proposals never alter cell values
            if prop.tier == "FLAG":
                continue

            for change in prop.changes:
                cell_key = (change.row, change.column)
                if cell_key in modified_cells:
                    # Invariant: cell touched by at most one proposal
                    continue

                r = change.row
                col = change.column
                if r in cleaned_df.index and col in cleaned_df.columns:
                    current_val = str(cleaned_df.at[r, col])
                    # Verify original value matches expected before change
                    if current_val == change.old_value:
                        cleaned_df.at[r, col] = str(change.new_value)
                        modified_cells.add(cell_key)
                        diff_records.append({
                            "rule_id": prop.id,
                            "rule_kind": prop.kind,
                            "tier": prop.tier,
                            "row": int(r),
                            "column": col,
                            "old_value": change.old_value,
                            "new_value": change.new_value
                        })

        # Ensure all types remain string
        cleaned_df = cleaned_df.astype(str).fillna("")
        return cleaned_df, diff_records
