"""
Cleaner - Rule-Engine Tabular Data Cleaning Package
Motto: Fix what can be proven, flag what can't.
"""

from cleaner.io import load_table, save_table
from cleaner.engine import CleaningEngine

__all__ = ["load_table", "save_table", "CleaningEngine"]
