"""
tests/test_overhaul_verification.py
Comprehensive test suite verifying all scientific, algorithmic, and architectural fixes:
1. R6 Functional Dependency continuous float guard (prevents price -> seller_type bugs)
2. Postal code false-alarm guard on numeric columns (prevents kms_driven -> postal_code)
3. Quality score monotonicity (score strictly increases on dirty data)
4. Downstream ML task type detection (regression vs classification) and clean held-out evaluation
5. Leading delimiter space ingestion handling (UCI comma-space datasets)
6. Untouched byte identity preservation guarantee
"""

import pytest
import pandas as pd
import numpy as np
import io

from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine
from cleaner.rules.r6_dependency_repair import propose_r6_dependency_repair, is_continuous_numeric
from cleaner.validators import check_universal_validators
from src.profiling.quality_metrics import compute_quality_score
from src.detection.anomaly_detector import detect_all_issues
from src.evaluation.downstream_evaluation import evaluate_downstream_ml, infer_target_column


def test_r6_continuous_numeric_guard():
    """Ensure continuous floats with small duplicate sets are NEVER used as determinant keys."""
    data = {
        "price": ["3.35", "4.75", "7.25", "2.85", "4.75", "3.35", "9.50", "12.30", "4.75"],
        "category": ["A", "B", "C", "A", "B", "A", "C", "C", "B"]
    }
    df = pd.DataFrame(data)
    assert is_continuous_numeric(df["price"]) is True

    proposals = propose_r6_dependency_repair(df)
    # price should not be determinant A
    price_as_determinant = [p for p in proposals if p.columns[0] == "price"]
    assert len(price_as_determinant) == 0


def test_postal_code_validator_ignores_generic_numeric():
    """Ensure numeric quantities like Kms_Driven or Salary are not falsely flagged as postal codes."""
    data = {
        "Kms_Driven": ["27000", "43000", "6900", "5200", "42450", "2071", "18796", "33429", "20273", "42367", "2135"],
        "Salary": ["50000", "65000", "72000", "120000", "48000", "55000", "80000", "95000", "60000", "52000", "45000"]
    }
    df = pd.DataFrame(data)
    proposals = check_universal_validators(df)
    postal_flags = [p for p in proposals if "postal_code" in p.id]
    assert len(postal_flags) == 0


def test_quality_score_monotonicity_on_dirty_data():
    """Ensure quality score strictly increases after cleaning dirty data (no score inversion)."""
    raw_df = load_table("sample_data/dirty_customer_dataset.csv")
    raw_issues = detect_all_issues(raw_df)
    raw_score = compute_quality_score(raw_df, detected_issues=raw_issues)["overall_score"]

    prof = profile_table(raw_df)
    engine = CleaningEngine()
    proposals = engine.generate_proposals(raw_df, prof)
    approved = [p for p in proposals if p.tier != "FLAG"]
    cleaned_df, _ = engine.apply(raw_df, approved)

    after_issues = detect_all_issues(cleaned_df)
    after_score = compute_quality_score(cleaned_df, detected_issues=after_issues)["overall_score"]

    assert after_score > raw_score, f"Expected after_score ({after_score}) > raw_score ({raw_score})"


def test_downstream_ml_regression_detection_and_evaluation():
    """Ensure continuous targets like car Selling_Price are recognized as regression."""
    raw_df = load_table("sample_data/car_dekho/car data.csv")
    prof = profile_table(raw_df)
    engine = CleaningEngine()
    proposals = engine.generate_proposals(raw_df, prof)
    approved = [p for p in proposals if p.tier != "FLAG"]
    cleaned_df, _ = engine.apply(raw_df, approved)

    res = evaluate_downstream_ml(raw_df, cleaned_df, target_column="Selling_Price", random_seed=42)
    assert res["status"] == "success"
    assert res["task_type"] == "Regression"
    assert "r2_score" in res["cleaned_metrics"]
    assert "rmse" in res["cleaned_metrics"]


def test_downstream_ml_classification_lift():
    """Ensure downstream classification accurately evaluates accuracy and F1-score on customer churn."""
    raw_df = load_table("sample_data/dirty_customer_dataset.csv")
    prof = profile_table(raw_df)
    engine = CleaningEngine()
    proposals = engine.generate_proposals(raw_df, prof)
    approved = [p for p in proposals if p.tier != "FLAG"]
    cleaned_df, _ = engine.apply(raw_df, approved)

    res = evaluate_downstream_ml(raw_df, cleaned_df, target_column="Churn", random_seed=42)
    assert res["status"] == "success"
    assert res["task_type"] == "Classification"
    assert res["cleaned_metrics"]["accuracy"] >= 0.70
    assert res["cleaned_metrics"]["f1_score"] >= 0.60


def test_csv_delimiter_space_stripping():
    """Ensure CSV lines with comma-space (', ') do not retain spurious delimiter spaces."""
    csv_text = "age, workclass, education\n39, State-gov, Bachelors\n50, Self-emp, Masters\n"
    buffer = io.StringIO(csv_text)
    df = load_table(buffer)
    assert df.at[0, "workclass"] == "State-gov"
    assert df.at[0, "education"] == "Bachelors"


def test_untouched_cell_byte_identity():
    """Ensure untouched cells remain 100% byte identical with zero silent type coercion."""
    data = {
        "id": ["00123", "00456", "00789"],
        "phone": ["+1-800-555-0199", "+1-800-555-0198", "+1-800-555-0197"],
        "text": [" clean text ", "good", "valid"]
    }
    df = pd.DataFrame(data)
    engine = CleaningEngine()
    prof = profile_table(df)
    proposals = engine.generate_proposals(df, prof)
    # Only approve text whitespace rule
    approved = [p for p in proposals if "text" in p.id]
    cleaned_df, diffs = engine.apply(df, approved)

    assert cleaned_df.at[0, "id"] == "00123"
    assert cleaned_df.at[1, "id"] == "00456"
    assert cleaned_df.at[0, "phone"] == "+1-800-555-0199"
    assert cleaned_df.at[0, "text"] == "clean text"
