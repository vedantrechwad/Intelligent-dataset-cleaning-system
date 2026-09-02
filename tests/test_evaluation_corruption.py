import pytest
import pandas as pd
import numpy as np
from src.evaluation.corruption_engine import DataCorruptionEngine
from src.evaluation.detection_metrics import evaluate_detection_performance
from src.evaluation.correction_metrics import evaluate_correction_performance
from src.evaluation.downstream_evaluation import evaluate_downstream_ml
from src.detection.anomaly_detector import detect_all_issues

def test_corruption_engine():
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Salary": [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000],
        "City": ["Mumbai", "Delhi", "Bangalore", "Pune", "Chennai", "Kolkata", "Hyderabad", "Jaipur", "Ahmedabad", "Surat"],
        "Join_Date": ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-06-01", "2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01"]
    })
    corrupter = DataCorruptionEngine(seed=42)
    corrupted_df, ground_truth = corrupter.inject_corruptions(
        df,
        missing_rate=0.1,
        duplicate_count=1,
        outlier_count=1,
        invalid_numeric_count=1,
        invalid_date_count=1,
        spelling_typo_count=1,
        casing_typo_count=1
    )
    assert len(ground_truth) > 0
    assert len(corrupted_df) >= len(df)

    # Test detection metrics calculation
    detected = detect_all_issues(corrupted_df)
    det_metrics = evaluate_detection_performance(ground_truth, detected)
    assert "overall" in det_metrics
    assert det_metrics["overall"]["true_positives"] >= 1
    assert "precision" in det_metrics["overall"]
    assert "recall" in det_metrics["overall"]

def test_downstream_ml_evaluation():
    dirty_df = pd.DataFrame({
        "feature1": [1, 2, np.nan, 4, 100, 6, 7, 8, 9, 10],
        "feature2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    clean_df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    res = evaluate_downstream_ml(dirty_df, clean_df, target_column="target", random_seed=42)
    assert "task_type" in res
    assert "corrupted_metrics" in res
    assert "cleaned_metrics" in res
