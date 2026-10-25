"""
tests/test_human_review_and_intelligence.py
Unit tests for Review Changes manual overrides, contextual assistant, and natural language rules.
"""

import pytest
import pandas as pd
from cleaner.engine import CleaningEngine
from cleaner.rules.base import Proposal, CellChange
from src.intelligence.contextual_assistant import explain_and_suggest_fix
from src.intelligence.nl_rule_generator import generate_rule_from_natural_language


def test_manual_overrides_application():
    df = pd.DataFrame({
        "Name": ["Alice", "J0hn Doe", "Bob"],
        "Age": ["25", "30", "40"],
        "DOB": ["1999-01-01", "1892-05-12", "1984-07-20"]
    })

    engine = CleaningEngine()
    manual_overrides = [
        {"row": 1, "column": "Name", "new_value": "John Doe"},
        {"row": 1, "column": "DOB", "new_value": "1992-05-12"}
    ]

    cleaned_df, diffs = engine.apply(df, [], manual_overrides=manual_overrides)

    assert cleaned_df.at[1, "Name"] == "John Doe"
    assert cleaned_df.at[1, "DOB"] == "1992-05-12"
    # Verify untouched cells are 100% identical
    assert cleaned_df.at[0, "Name"] == "Alice"
    assert cleaned_df.at[2, "Name"] == "Bob"
    assert cleaned_df.at[0, "DOB"] == "1999-01-01"

    manual_diffs = [d for d in diffs if d["rule_id"] == "MANUAL_EDIT"]
    assert len(manual_diffs) == 2


def test_contextual_assistant_dob_typo():
    row_ctx = {
        "Name": "John Doe",
        "HireDate": "2018-05-01",
        "Department": "Engineering",
        "DOB": "1892-05-12"
    }

    res = explain_and_suggest_fix(row_ctx, "DOB", "1892-05-12", "Suspicious century in date field", ollama_client=None)

    assert res["status"] if "status" in res else True
    assert "century typo" in res["explanation"].lower() or "dob" in res["explanation"].lower()
    assert res["suggested_value"] == "1992-05-12"


def test_contextual_assistant_name_with_digits():
    row_ctx = {
        "Name": "J0hn Sm1th",
        "Role": "Developer",
        "Location": "New York"
    }

    res = explain_and_suggest_fix(row_ctx, "Name", "J0hn Sm1th", "Name contains digit characters", ollama_client=None)

    assert "John Smith" in res["suggested_value"]


def test_natural_language_rule_generator_heuristic():
    df = pd.DataFrame({
        "Total Spent": [100.0, -15.0, 50.0],
        "Quantity": [2, 1, 3]
    })

    res = generate_rule_from_natural_language(df, "Total Spent must be greater than 0", ollama_client=None)

    assert res["status"] == "success"
    assert 1 in res["violating_row_indices"]
    assert len(res["violating_row_indices"]) == 1


def test_metric_explainer_shifts():
    from src.intelligence.metric_explainer import explain_metric_shift

    # 1. Quality Score Improvement
    exp1 = explain_metric_shift("Quality Score", 83.5, 90.95, applied_rules=["WhitespaceTrim", "SentinelNulls"], dataset_context={"name": "Adult Census", "rows": 32560})
    assert exp1["direction"] == "improved"
    assert "Overall Quality Score" in exp1["headline"]
    assert len(exp1["explanation"]) > 30

    # 2. Completeness Reduction (Disguised Sentinels Revealed)
    exp2 = explain_metric_shift("Completeness", 99.1, 92.4, applied_rules=["ConvertSentinelsToNull"])
    assert exp2["direction"] == "reduced"
    assert "sentinel" in exp2["explanation"].lower() or "missingness" in exp2["explanation"].lower()

    # 3. Downstream ML R2 Improvement
    exp3 = explain_metric_shift("R-Squared (ML Utility)", 0.003, 0.595, applied_rules=["TypeCast", "DropUnitNoise"])
    assert exp3["direction"] == "improved"
    assert "R²" in exp3["headline"] or "r2" in exp3["headline"].lower()

    # 4. Downstream ML R2 Minor Dip (Outlier Winsorization)
    exp4 = explain_metric_shift("R-Squared (ML Utility)", 0.113, 0.076, applied_rules=["WinsorizeOutliers"])
    assert exp4["direction"] == "reduced"
    assert "outlier" in exp4["explanation"].lower() or "variance" in exp4["explanation"].lower()

