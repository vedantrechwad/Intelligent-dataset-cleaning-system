"""
cleaner/rules/base.py
Core Proposal model and rule ordering specifications.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple


RULE_ORDER = [
    "R1_missing_tokens",
    "R2_numeric",
    "R3_whitespace_case",
    "R4_dates",
    "R5_compound_split",
    "R6_dependency_repair",
    "R7_category_variants",
    "R8_exact_duplicates"
]

RULE_ORDER_MAP = {name: idx for idx, name in enumerate(RULE_ORDER)}


@dataclass
class CellChange:
    row: int
    column: str
    old_value: str
    new_value: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row": self.row,
            "column": self.column,
            "old_value": self.old_value,
            "new_value": self.new_value
        }


@dataclass
class Proposal:
    id: str
    kind: str
    tier: str  # "AUTO", "REVIEW", or "FLAG"
    columns: List[str]
    description: str
    evidence: str
    changes: List[CellChange] = field(default_factory=list)
    dropped_rows: List[int] = field(default_factory=list)
    n_cells: int = 0
    sample: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.n_cells:
            self.n_cells = len(self.changes) if self.changes else len(self.dropped_rows)
        if not self.sample and self.changes:
            self.sample = [c.to_dict() for c in self.changes[:5]]
        elif not self.sample and self.dropped_rows:
            self.sample = [{"dropped_row_index": r} for r in self.dropped_rows[:5]]

    @property
    def order_rank(self) -> int:
        return RULE_ORDER_MAP.get(self.kind, 999)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "tier": self.tier,
            "columns": self.columns,
            "description": self.description,
            "evidence": self.evidence,
            "n_cells": self.n_cells,
            "sample": self.sample,
            "changes": [c.to_dict() for c in self.changes],
            "dropped_rows": list(self.dropped_rows)
        }
