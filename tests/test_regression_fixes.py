import pytest
import pandas as pd
import numpy as np

from src.detection.missing_detector import detect_missing_values
from src.detection.inconsistency_detector import detect_categorical_inconsistencies
from src.detection.type_validator import detect_type_inconsistencies
from src.detection.date_validator import validate_dates
from src.detection.range_validator import validate_domain_ranges
from src.cleaning.pipeline import execute_cleaning_pipeline

def get_routing_decisions(issues):
    from src.intelligence.confidence_engine import route_decision
    for issue in issues:
        issue['routing_decision'] = route_decision(issue.get("correction_confidence", 0.0), issue.get("issue_type"))
    return issues

def test_safe_automatic_formatting():
    """Tests 1-8: Safe automatic normalization (whitespace, casing, obvious typos)"""
    df = pd.DataFrame({
        "City": [" Mumbai ", "mumbai", "mumbai", "MALE", "Male", "Male", "yes", "Yes", "Yes", "Femlae", "Female", "Female", "Mlae", "Male", "Male", "Yse", "Yes", "Yes", "Month-to-mont", "Month-to-month", "Month-to-month", "Eletronic check", "Electronic check", "Electronic check"]
    })
    
    issues = detect_categorical_inconsistencies(df, similarity_threshold=82.0)
    issues = get_routing_decisions(issues)
    
    cleaned_df, logs = execute_cleaning_pipeline(df, issues, cleaning_options={"remove_exact_duplicates": False})
    
    assert cleaned_df.at[0, "City"] == "mumbai" # 1
    assert cleaned_df.at[3, "City"] == "Male" # 2 
    assert cleaned_df.at[6, "City"] == "Yes"  # 3
    assert cleaned_df.at[9, "City"] == "Female" # 4 (Femlae -> Female)
    assert cleaned_df.at[12, "City"] == "Male" # 5 (Mlae -> Male)
    assert cleaned_df.at[15, "City"] == "Yes" # 6 (Yse -> Yes)
    assert cleaned_df.at[18, "City"] == "Month-to-month" # 7
    assert cleaned_df.at[21, "City"] == "Electronic check" # 8

def test_safe_numeric_and_duplicates():
    """Tests 9-12: Numeric strings, duplicates, NULL, N/A"""
    df = pd.DataFrame({
        "Age": ["25", "25", "30", "30", "40", "40", "50", "NULL", "N/A"]
    })
    
    # 9, 11, 12
    type_issues = detect_type_inconsistencies(df)
    type_issues = get_routing_decisions(type_issues)
    
    cleaned_df, logs = execute_cleaning_pipeline(
        df, type_issues, 
        cleaning_options={"remove_exact_duplicates": True, "impute_missing": False}
    )
    
    assert len(cleaned_df) == 5 # 10 (exact duplicate removed)
    assert cleaned_df.at[0, "Age"] == 25 # 9 (string 25 to numeric 25)
    
    # NULL and N/A should be converted to NaN
    assert pd.isna(cleaned_df.at[4, "Age"]) # 11 & 12

def test_must_not_autocorrect():
    """Tests 13-24: Invalid inputs that MUST have 0 correction confidence"""
    # 13, 14
    df_age = pd.DataFrame({"Age": [-5, 250, 30]})
    range_issues = validate_domain_ranges(df_age)
    assert len(range_issues) == 2
    assert all(i["correction_confidence"] == 0.0 for i in range_issues)

    # 16
    df_date = pd.DataFrame({"Joined": ["2025-02-30"]})
    date_issues = validate_dates(df_date)
    assert len(date_issues) == 1
    assert date_issues[0]["correction_confidence"] == 0.0
    
    # 18: Unknown category with multiple candidates
    df_ambiguous = pd.DataFrame({
        "Category": ["Washngton", "Washington", "Washington", "Washinton", "Washinton"]
    })
    inc_issues = detect_categorical_inconsistencies(df_ambiguous)
    ambig_issue = [i for i in inc_issues if i["original_value"] == "Washngton"][0]
    # Ensure it's not Auto_correct
    assert ambig_issue["correction_confidence"] < 0.95

def test_semantic_safety():
    """Tests 25-29: Semantic distinctions and strict missing boundaries"""
    df = pd.DataFrame({
        "Internet": ["No", "No", "No", "No internet service", "No internet service"],
        "Partner": ["Unknown", "Yes", "No", "Yes", "No"]
    })
    
    # 25. "No internet service" vs "No" should NOT be flagged as spelling typo
    inc_issues = detect_categorical_inconsistencies(df)
    assert len(inc_issues) == 0 # No typos detected between the two
    
    # 26. "Unknown" must not automatically become a missing value
    missing = detect_missing_values(df)
    assert len(missing) == 0 # "Unknown" is NOT in missing_detector list anymore!
    
    # 27, 28, 29. Missing categories should not be blindly guessed
    df_miss = pd.DataFrame({
        "Gender": ["Male", "Male", "Female", "NULL"]
    })
    type_issues = detect_type_inconsistencies(df_miss)
    cleaned_df, logs = execute_cleaning_pipeline(
        df_miss, 
        type_issues, 
        cleaning_options={"missing_categorical_strategy": "skip", "remove_exact_duplicates": False}
    )
    
    # "skip" means the NaN remains NaN
    assert pd.isna(cleaned_df.at[3, "Gender"])
