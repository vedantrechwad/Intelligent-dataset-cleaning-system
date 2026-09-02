from typing import Dict, Any, Optional
from src.utils.helpers import load_config

def calculate_category_correction_confidence(
    similarity_score: float, # 0 to 100
    dominant_frequency: int,
    candidate_frequency: int,
    llm_confidence: Optional[float] = None
) -> float:
    """
    Computes calibrated correction confidence for categorical typos.
    Formula:
      confidence = 0.50 * similarity + 0.30 * frequency_evidence + 0.20 * contextual_evidence
    """
    sim_component = similarity_score / 100.0

    # Frequency evidence: higher when dominant category overwhelmingly outnumbers the typo
    total_freq = max(1, dominant_frequency + candidate_frequency)
    freq_ratio = (dominant_frequency - candidate_frequency) / total_freq
    freq_component = max(0.0, min(1.0, freq_ratio))

    # Contextual evidence (LLM if available, else penalty if candidate occurs more than once)
    if llm_confidence is not None:
        context_component = llm_confidence
    else:
        context_component = 0.85 if candidate_frequency == 1 else 0.50

    confidence = (0.50 * sim_component) + (0.30 * freq_component) + (0.20 * context_component)
    return round(float(max(0.0, min(0.99, confidence))), 3)


def calculate_outlier_confidence(
    num_signals: int,
    max_zscore: float,
    iqr_distance_ratio: float,
    isolation_score: float
) -> float:
    """
    Combines outlier detection signals into a normalized anomaly confidence score.
    """
    z_signal = min(1.0, max_zscore / 5.0) if max_zscore > 0 else 0.0
    iqr_signal = min(1.0, iqr_distance_ratio / 3.0)
    iso_signal = min(1.0, isolation_score)

    weights = [0.35, 0.35, 0.30]
    conf = (weights[0] * z_signal) + (weights[1] * iqr_signal) + (weights[2] * iso_signal)
    
    # Boost if multiple independent detectors agree
    if num_signals >= 3:
        conf = min(0.98, conf * 1.25)
    elif num_signals == 2:
        conf = min(0.92, conf * 1.10)

    return round(float(max(0.50, min(0.99, conf))), 3)


def route_decision(confidence: float, issue_type: str, requires_human: bool = False) -> str:
    """
    Maps correction confidence and issue nature to decision:
      AUTO_CORRECT (>= 0.95)
      SUGGEST_REVIEW (0.70 <= c < 0.95)
      HUMAN_REVIEW_REQUIRED (< 0.70 or explicit domain constraint)
    """
    if requires_human:
        return "HUMAN_REVIEW_REQUIRED"

    cfg = load_config().get("thresholds", {})
    auto_thresh = cfg.get("auto_correct_confidence", 0.95)
    suggest_thresh = cfg.get("suggest_review_confidence", 0.70)

    if confidence >= auto_thresh:
        return "AUTO_CORRECT"
    elif confidence >= suggest_thresh:
        return "SUGGEST_REVIEW"
    else:
        return "HUMAN_REVIEW_REQUIRED"
