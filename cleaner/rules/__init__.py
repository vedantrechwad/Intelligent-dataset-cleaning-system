"""
cleaner/rules package
"""

from cleaner.rules.base import Proposal, CellChange, RULE_ORDER
from cleaner.rules.r1_missing import propose_r1_missing
from cleaner.rules.r2_numeric import propose_r2_numeric
from cleaner.rules.r3_whitespace_case import propose_r3_whitespace_case

__all__ = [
    "Proposal",
    "CellChange",
    "RULE_ORDER",
    "propose_r1_missing",
    "propose_r2_numeric",
    "propose_r3_whitespace_case",
]
