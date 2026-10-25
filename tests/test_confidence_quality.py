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


def test_defect_resolution_rate_metric():
    from src.profiling.quality_metrics import compute_defect_resolution_rate, compute_quality_score
    drr = compute_defect_resolution_rate(initial_defects=500, repaired_defects=450)
    assert drr["resolution_rate_pct"] == 90.0
    assert drr["remaining_defects"] == 50

    # DRR in quality score
    df = pd.DataFrame({"A": [1, 2, 3]})
    score = compute_quality_score(df, initial_issue_count=100, defects_repaired=85)
    assert score["defect_resolution_rate"] is not None
    assert score["defect_resolution_rate"]["resolution_rate_pct"] == 85.0


def test_continuous_penalty_no_clipping():
    """Verify that heavy inconsistencies do not prematurely clip score to 0.0 and repairs always produce score lift."""
    from src.profiling.quality_metrics import compute_quality_score
    
    # 50 rows, 2 columns = 100 cells
    df = pd.DataFrame({
        "A": [f"val_{i}" for i in range(50)],
        "B": [f"text_{i}" for i in range(50)]
    })
    
    # 30 inconsistencies (30% error rate) - previously clamped to 0.0 by 10x multiplier
    issues_30 = [{"issue_type": "casing_issue"} for _ in range(30)]
    score_30 = compute_quality_score(df, detected_issues=issues_30)
    assert score_30["dimensions"]["consistency"] > 0.0
    
    # 15 inconsistencies (repair 15 defects)
    issues_15 = [{"issue_type": "casing_issue"} for _ in range(15)]
    score_15 = compute_quality_score(df, detected_issues=issues_15)
    
    # Strictly monotonic score lift
    assert score_15["dimensions"]["consistency"] > score_30["dimensions"]["consistency"]
    assert score_15["overall_score"] > score_30["overall_score"]

