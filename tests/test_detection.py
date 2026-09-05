import pytest
import pandas as pd
import numpy as np
from src.detection.missing_detector import detect_missing_values
from src.detection.duplicate_detector import detect_exact_duplicates, detect_near_duplicates
from src.detection.range_validator import validate_domain_ranges
from src.detection.date_validator import validate_dates
from src.detection.type_validator import detect_type_inconsistencies
from src.detection.outlier_detector import detect_numerical_outliers
from src.detection.inconsistency_detector import detect_categorical_inconsistencies
from src.detection.anomaly_detector import detect_all_issues

def test_missing_detection():
    df = pd.DataFrame({
        "A": [1, np.nan, "null", "N/A", "unknown", "valid"]
    })
    issues = detect_missing_values(df)
    assert len(issues) == 3
    for issue in issues:
        assert issue["issue_type"] == "missing_value"

def test_exact_and_near_duplicates():
    df = pd.DataFrame({
        "id": [1, 2, 1, 3],
        "name": ["Alice", "Bob", "Alice", "Bobb"]
    })
    exact = detect_exact_duplicates(df)
    assert len(exact) == 1
    assert exact[0]["row"] == 2

    near = detect_near_duplicates(df, similarity_threshold=80.0)
    assert len(near) >= 1

def test_domain_range_validation():
    df = pd.DataFrame({
        "Age": [25, -5, 140, 30],
        "Percentage": [50, -10, 105, 90]
    })
    issues = validate_domain_ranges(df)
    assert len(issues) == 4
    # Ensure correction_confidence is 0.0 for human review
    for issue in issues:
        assert issue["is_human_review_required"] is True
        assert issue["correction_confidence"] == 0.0

def test_date_validation():
    df = pd.DataFrame({
        "Join_Date": ["2024-01-15", "22/13/2020", "2025-02-30", "2023-08-19"]
    })
    issues = validate_dates(df)
    assert len(issues) == 2
    for issue in issues:
        assert issue["is_human_review_required"] is True

def test_categorical_inconsistency_detection():
    df = pd.DataFrame({
        "City": ["Mumbai", "mumbai", "MUMBAI", " Mumbai", "Mumabi", "Mumbai", "Mumbai", "Delhi"]
    })
    issues = detect_categorical_inconsistencies(df)
    issue_types = {i["issue_type"] for i in issues}
    assert "whitespace_inconsistency" in issue_types
    assert "casing_inconsistency" in issue_types
    assert "spelling_typo" in issue_types

def test_outlier_detection():
    # Extreme outlier
    df = pd.DataFrame({
        "Salary": [50000, 52000, 48000, 51000, 49000, 53000, 50500, 49500, 51500, 50000, 1500000]
    })
    issues = detect_numerical_outliers(df)
    assert len(issues) >= 1
    assert issues[0]["original_value"] == 1500000
