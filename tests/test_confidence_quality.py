import pytest
import pandas as pd
from src.intelligence.confidence_engine import calculate_category_correction_confidence, route_decision
from src.profiling.quality_metrics import compute_quality_score

def test_confidence_calculation():
    conf_high = calculate_category_correction_confidence(
        similarity_score=95.0,
        dominant_frequency=50,
        candidate_frequency=1,
        llm_confidence=0.98
    )
    assert conf_high >= 0.95

    conf_low = calculate_category_correction_confidence(
        similarity_score=75.0,
        dominant_frequency=3,
        candidate_frequency=2,
        llm_confidence=0.50
    )
    assert conf_low < 0.85

def test_route_decision():
    assert route_decision(0.98, "spelling_typo") == "AUTO_CORRECT"
    assert route_decision(0.85, "spelling_typo") == "SUGGEST_REVIEW"
    assert route_decision(0.50, "spelling_typo") == "HUMAN_REVIEW_REQUIRED"
    assert route_decision(0.99, "invalid_range", requires_human=True) == "HUMAN_REVIEW_REQUIRED"

def test_quality_score_dimensions():
    df_clean = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": ["X", "Y", "Z", "W"]
    })
    score_clean = compute_quality_score(df_clean)
    assert score_clean["overall_score"] == 100.0

    df_dirty = pd.DataFrame({
        "A": [1, None, None, 4],
        "B": ["X", "X", "X", "X"]
    })
    score_dirty = compute_quality_score(df_dirty)
    assert score_dirty["overall_score"] < 100.0
    assert score_dirty["dimensions"]["completeness"] < 100.0
