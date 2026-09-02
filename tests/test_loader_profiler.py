import os
import pytest
import pandas as pd
import numpy as np
from src.ingestion.file_loader import load_dataset
from src.ingestion.schema_detector import detect_column_schema, detect_dataset_schema
from src.profiling.profiler import profile_dataset
from src.profiling.quality_metrics import compute_quality_score

SAMPLE_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "dirty_customer_dataset.csv")
SAMPLE_XLSX = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "dirty_customer_dataset.xlsx")

def test_load_csv_dataset():
    orig_df, work_df, meta = load_dataset(SAMPLE_CSV)
    assert len(orig_df) == 25
    assert len(work_df) == 25
    assert meta["format"] == "CSV"
    assert "Customer_ID" in meta["columns"]
    # Verify immutability
    work_df.at[0, "Age"] = 99
    assert orig_df.at[0, "Age"] != 99

def test_load_xlsx_dataset():
    orig_df, work_df, meta = load_dataset(SAMPLE_XLSX)
    assert len(orig_df) == 25
    assert meta["format"] == "XLSX"

def test_schema_detection():
    df = pd.DataFrame({
        "num": [10, 20, 30, 40],
        "cat": ["A", "B", "A", "B"],
        "date_col": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]
    })
    schema = detect_dataset_schema(df)
    assert schema["num"]["logical_type"] == "numeric"
    assert schema["cat"]["logical_type"] == "categorical"
    assert schema["date_col"]["logical_type"] == "datetime"

def test_profiler_and_quality_score():
    df = pd.DataFrame({
        "Age": [25, 30, np.nan, 40],
        "City": ["Mumbai", "Delhi", "Mumbai", "Pune"]
    })
    profile = profile_dataset(df)
    assert profile["row_count"] == 4
    assert profile["column_count"] == 2
    assert profile["missing_cells"] == 1
    assert profile["columns"]["Age"]["missing_count"] == 1
    assert profile["columns"]["Age"]["mean"] == pytest.approx(31.667, 0.01)

    score_res = compute_quality_score(df)
    assert score_res["overall_score"] > 0
    assert "completeness" in score_res["dimensions"]
