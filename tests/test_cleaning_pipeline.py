import pytest
import pandas as pd
import numpy as np
from src.cleaning.duplicate_cleaner import clean_exact_duplicates
from src.cleaning.category_cleaner import clean_categorical_inconsistencies
from src.cleaning.type_cleaner import clean_data_types
from src.cleaning.missing_cleaner import clean_missing_values
from src.cleaning.pipeline import execute_cleaning_pipeline
from src.detection.anomaly_detector import detect_all_issues

def test_duplicate_cleaner():
    df = pd.DataFrame({
        "A": [1, 2, 1, 3],
        "B": ["x", "y", "x", "z"]
    })
    clean_df, logs = clean_exact_duplicates(df)
    assert len(clean_df) == 3
    assert len(logs) == 1

def test_category_cleaner():
    df = pd.DataFrame({
        "City": [" Mumbai ", "mumbai", "MUMBAI", "Mumbai", "Delhi"]
    })
    clean_df, logs = clean_categorical_inconsistencies(df)
    assert clean_df["City"].tolist() == ["Mumbai", "Mumbai", "Mumbai", "Mumbai", "Delhi"]
    assert len(logs) >= 3

def test_missing_cleaner():
    df = pd.DataFrame({
        "Age": [20, 30, np.nan, 40],
        "City": ["Mumbai", "Delhi", "null", "Mumbai"]
    })
    clean_df, logs = clean_missing_values(df, numeric_strategy="median", categorical_strategy="mode")
    assert clean_df["Age"].isna().sum() == 0
    assert clean_df.at[2, "Age"] == 30.0
    assert clean_df.at[2, "City"] == "Mumbai"

def test_end_to_end_cleaning_pipeline():
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Alice", "Charlie"],
        "Age": ["25", "-5", "25", "35"],
        "City": [" Mumbai", "Delhi", " Mumbai", "delhi"]
    })
    issues = detect_all_issues(df)
    # Provide human correction for row 1 invalid age -5
    user_corrections = {
        "ISSUE_CUSTOM": {
            "status": "accepted",
            "row": 1,
            "column": "Age",
            "original_value": "-5",
            "corrected_value": 30,
            "issue_type": "invalid_range"
        }
    }
    clean_df, audit = execute_cleaning_pipeline(df, issues, user_corrections=user_corrections)
    assert len(clean_df) == 3 # duplicate removed
    assert clean_df.at[1, "Age"] == 30
    assert len(audit) > 0
