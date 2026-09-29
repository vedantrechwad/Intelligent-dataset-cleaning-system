"""
cleaner/rules package
"""

from cleaner.rules.base import Proposal, CellChange, RULE_ORDER
from cleaner.rules.r1_missing import propose_r1_missing
from cleaner.rules.r2_numeric import propose_r2_numeric
from cleaner.rules.r3_whitespace_case import propose_r3_whitespace_case
from cleaner.rules.r5_compound_split import propose_r5_compound_split
from cleaner.rules.r6_dependency_repair import propose_r6_dependency_repair
from cleaner.rules.r8_exact_duplicates import propose_r8_exact_duplicates

__all__ = [
    "Proposal",
    "CellChange",
    "RULE_ORDER",
    "propose_r1_missing",
    "propose_r2_numeric",
    "propose_r3_whitespace_case",
    "propose_r5_compound_split",
    "propose_r6_dependency_repair",
    "propose_r8_exact_duplicates",
]
