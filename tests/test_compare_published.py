"""
tests/test_compare_published.py
Unit tests for eval/compare_published.py metrics and oracle detection filter.
"""

import pandas as pd
import pytest
from eval.compare_published import (
    calculate_metrics,
    apply_oracle_detection_filter,
    eq_exact,
    eq_numeric
)


def test_eq_functions():
    assert eq_exact("hello", "hello")
    assert not eq_exact("hello", "world")
    assert not eq_exact("12.0", "12")

    assert eq_numeric("12.0", "12")
    assert eq_numeric("12.000", "12.0")
    assert not eq_numeric("12.5", "12.0")
    assert eq_numeric("abc", "abc")
    assert not eq_numeric("abc", "def")


def test_metrics_on_handcrafted_example():
    """
    Test metric calculation on a hand-crafted 5-cell table:
    Row 0: Clean cell, left untouched (Clean -> Clean -> Clean)
    Row 1: Clean cell, damaged by cleaner (Clean -> Clean -> Damaged)
    Row 2: Error cell, successfully fixed (Dirty -> Clean -> Clean)
    Row 3: Error cell, missed/untouched (Dirty -> Clean -> Dirty)
    Row 4: Error cell, wrongly fixed (Dirty -> Clean -> Wrong)
    """
    df_dirty = pd.DataFrame({"val": ["A", "B", "C_dirty", "D_dirty", "E_dirty"]})
    df_clean = pd.DataFrame({"val": ["A", "B", "C_clean", "D_clean", "E_clean"]})
    df_out   = pd.DataFrame({"val": ["A", "B_damaged", "C_clean", "D_dirty", "E_wrong"]})

    # Exact metrics
    # True errors (|E|): rows 2, 3, 4 -> 3 errors
    # Clean cells: rows 0, 1 -> 2 clean cells
    # Changed cells: row 1 (B->B_damaged), row 2 (C_dirty->C_clean), row 4 (E_dirty->E_wrong) -> 3 changed
    # Fixed: row 2 -> 1 fixed
    # Damaged: row 1 -> 1 damaged
    # Recall = fixed / |E| = 1 / 3 = 0.3333...
    # Precision = fixed / changed = 1 / 3 = 0.3333...
    # F1 = 1 / 3 = 0.3333...
    # Harm rate = damaged / clean cells = 1 / 2 = 0.50
    m = calculate_metrics(df_dirty, df_clean, df_out, numeric_tolerant=False)

    assert m["E_count"] == 3
    assert m["clean_cells_count"] == 2
    assert m["fixed"] == 1
    assert m["cells_changed"] == 3
    assert m["damaged"] == 1
    assert pytest.approx(m["recall"], 1e-4) == 1.0 / 3.0
    assert pytest.approx(m["precision"], 1e-4) == 1.0 / 3.0
    assert pytest.approx(m["f1"], 1e-4) == 1.0 / 3.0
    assert pytest.approx(m["harm_rate"], 1e-4) == 0.50


def test_metrics_numeric_tolerance():
    df_dirty = pd.DataFrame({"num": ["10.0 oz", "20"]})
    df_clean = pd.DataFrame({"num": ["10", "20"]})
    df_out   = pd.DataFrame({"num": ["10.0", "20"]})

    # Exact: "10.0" != "10", so not fixed
    m_exact = calculate_metrics(df_dirty, df_clean, df_out, numeric_tolerant=False)
    assert m_exact["fixed"] == 0

    # Numeric tolerant: "10.0" == 10.0 == "10", so fixed
    m_num = calculate_metrics(df_dirty, df_clean, df_out, numeric_tolerant=True)
    assert m_num["fixed"] == 1
    assert m_num["recall"] == 1.0
    assert m_num["precision"] == 1.0


def test_index_column_exclusion():
    # 2 columns: index, value
    df_dirty = pd.DataFrame({"index": ["0", "1"], "val": ["X_dirty", "Y_clean"]})
    df_clean = pd.DataFrame({"index": ["0", "1"], "val": ["X_clean", "Y_clean"]})
    df_out   = pd.DataFrame({"index": ["0", "1"], "val": ["X_clean", "Y_clean"]})

    # Exclude index col
    m = calculate_metrics(df_dirty, df_clean, df_out, exclude_first_col_as_index=True)
    assert m["E_count"] == 1  # Only row 0 val
    assert m["clean_cells_count"] == 1  # Only row 1 val
    assert m["fixed"] == 1
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_oracle_detection_never_changes_clean_cell():
    """
    Verify that oracle_detection reverts any changes on cells that were originally clean.
    """
    df_dirty = pd.DataFrame({
        "c1": ["good_1", "bad_1", "good_2"],
        "c2": ["bad_2", "good_3", "good_4"]
    })
    df_clean = pd.DataFrame({
        "c1": ["good_1", "fixed_1", "good_2"],
        "c2": ["fixed_2", "good_3", "good_4"]
    })
    # Cleaner made fixes, but also accidentally damaged good_1 and good_3
    df_out = pd.DataFrame({
        "c1": ["damaged_1", "fixed_1", "good_2"],
        "c2": ["fixed_2", "damaged_3", "good_4"]
    })

    df_oracle = apply_oracle_detection_filter(df_dirty, df_clean, df_out, numeric_tolerant=False)

    # Oracle must have reverted damaged_1 back to good_1, and damaged_3 back to good_3
    assert df_oracle.iat[0, 0] == "good_1"
    assert df_oracle.iat[1, 1] == "good_3"

    # And preserved true repairs
    assert df_oracle.iat[1, 0] == "fixed_1"
    assert df_oracle.iat[0, 1] == "fixed_2"

    # Metrics on oracle output must have damaged == 0 and harm_rate == 0
    m_oracle = calculate_metrics(df_dirty, df_clean, df_oracle, numeric_tolerant=False)
    assert m_oracle["damaged"] == 0
    assert m_oracle["harm_rate"] == 0.0
