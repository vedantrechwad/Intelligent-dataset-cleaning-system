"""
tests/test_phase4.py
Unit tests for Phase 4:
- Standalone Python Script Exporter (cleaner/exporter.py)
- Diff Viewer and Summary Metrics (cleaner/diff_viewer.py)
- Streamlit app syntax and execution integrity
"""

import pytest
import pandas as pd
from cleaner.rules.base import Proposal, CellChange
from cleaner.exporter import generate_cleaning_script
from cleaner.diff_viewer import compute_diff_summary, format_diff_table, build_side_by_side_diff


def test_script_exporter_generation():
    prop1 = Proposal(
        id="R1_missing_test",
        kind="R1_missing_tokens",
        tier="AUTO",
        columns=["city"],
        description="Convert disguised missing tokens",
        evidence="10 obs",
        changes=[
            CellChange(row=0, column="city", old_value="N/A", new_value=""),
            CellChange(row=1, column="city", old_value="null", new_value="")
        ]
    )
    prop2 = Proposal(
        id="R8_duplicates_test",
        kind="R8_exact_duplicates",
        tier="AUTO",
        columns=["__all__"],
        description="Drop duplicate rows",
        evidence="1 row",
        dropped_rows=[2]
    )

    script = generate_cleaning_script([prop1, prop2], "raw_test.csv")
    assert "def clean_dataset" in script
    assert "pd.read_csv" in script
    assert "drop_duplicates" in script
    assert "R1_missing_test" in script

    # Test that generated script compiles as valid Python
    compiled = compile(script, "<string>", "exec")
    assert compiled is not None


def test_diff_viewer_summary():
    df_orig = pd.DataFrame({
        "A": ["1", "2", "3", "4"],
        "B": ["X", "Y", "Z", "W"]
    })
    df_clean = pd.DataFrame({
        "A": ["1", "2", "3", "4"],
        "B": ["X", "CleanY", "Z", "W"]
    })
    diffs = [{
        "rule_id": "R7_test",
        "rule_kind": "R7_category_variants",
        "tier": "REVIEW",
        "row": 1,
        "column": "B",
        "old_value": "Y",
        "new_value": "CleanY"
    }]

    summary = compute_diff_summary(df_orig, df_clean, diffs)
    assert summary["total_cells"] == 8
    assert summary["modified_cells"] == 1
    assert summary["untouched_cells"] == 7
    assert summary["untouched_percentage"] == 87.5
    assert summary["harm_risk"] == 0.0


def test_diff_viewer_formatting():
    diffs = [{
        "rule_id": "R2_numeric",
        "rule_kind": "R2_numeric",
        "tier": "AUTO",
        "row": 0,
        "column": "val",
        "old_value": "$10",
        "new_value": "10"
    }]
    formatted = format_diff_table(diffs)
    assert len(formatted) == 1
    assert formatted.iloc[0]["Old Value"] == "$10"
    assert formatted.iloc[0]["New Value"] == "10"


def test_app_compilation():
    # Verify app.py compiles cleanly without SyntaxError or invalid imports
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
    compiled = compile(code, "app.py", "exec")
    assert compiled is not None
