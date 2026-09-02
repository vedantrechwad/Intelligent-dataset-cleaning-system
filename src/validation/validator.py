from typing import Tuple, List, Dict, Any
import pandas as pd
from src.profiling.profiler import profile_dataset
from src.profiling.quality_metrics import compute_quality_score
from src.detection.anomaly_detector import detect_all_issues

def validate_cleaned_dataset(
    cleaned_df: pd.DataFrame
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Re-runs the full profiling and detection pipeline on the cleaned dataset.
    Never assumes issues are solved without explicit verification.
    """
    profile = profile_dataset(cleaned_df)
    remaining_issues = detect_all_issues(cleaned_df)
    quality_score = compute_quality_score(cleaned_df, detected_issues=remaining_issues)

    return profile, remaining_issues, quality_score
