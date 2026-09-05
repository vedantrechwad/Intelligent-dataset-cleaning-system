from typing import List, Dict, Any
import pandas as pd
from src.detection.missing_detector import detect_missing_values
from src.detection.duplicate_detector import detect_exact_duplicates, detect_near_duplicates
from src.detection.range_validator import validate_domain_ranges
from src.detection.date_validator import validate_dates
from src.detection.type_validator import detect_type_inconsistencies
from src.detection.outlier_detector import detect_numerical_outliers
from src.detection.inconsistency_detector import detect_categorical_inconsistencies
from src.utils.helpers import load_config, logger

def detect_all_issues(df: pd.DataFrame, dynamic_rules: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Unified detection coordinator.
    Runs all specialized statistical, rule-based, and ML detectors.
    Deduplicates overlapping issues (e.g. invalid range takes precedence over statistical outlier).
    Assigns unique issue IDs and classifies routing decisions.
    """
    cfg = load_config()
    auto_thresh = cfg.get("thresholds", {}).get("auto_correct_confidence", 0.95)

    all_raw_issues = []

    # 1. Exact duplicates
    try:
        all_raw_issues.extend(detect_exact_duplicates(df))
    except Exception as e:
        logger.error(f"Error in exact duplicate detection: {e}")

    # 2. Near duplicates
    try:
        all_raw_issues.extend(detect_near_duplicates(df))
    except Exception as e:
        logger.error(f"Error in near duplicate detection: {e}")

    # 3. Missing values
    try:
        all_raw_issues.extend(detect_missing_values(df))
    except Exception as e:
        logger.error(f"Error in missing value detection: {e}")

    # 4. Range and domain rule violations (High priority validity)
    try:
        all_raw_issues.extend(validate_domain_ranges(df, custom_rules=dynamic_rules))
    except Exception as e:
        logger.error(f"Error in range validation: {e}")

    # 5. Invalid and unparseable dates (High priority validity)
    try:
        all_raw_issues.extend(validate_dates(df))
    except Exception as e:
        logger.error(f"Error in date validation: {e}")

    # 6. Type inconsistencies
    try:
        all_raw_issues.extend(detect_type_inconsistencies(df))
    except Exception as e:
        logger.error(f"Error in type validation: {e}")

    # 7. Categorical inconsistencies (whitespace, casing, spelling)
    try:
        all_raw_issues.extend(detect_categorical_inconsistencies(df))
    except Exception as e:
        logger.error(f"Error in categorical inconsistency detection: {e}")

    # 8. Statistical and ML Outliers
    try:
        all_raw_issues.extend(detect_numerical_outliers(df))
    except Exception as e:
        logger.error(f"Error in outlier detection: {e}")

    # Deduplicate issues for the same (row, col)
    # Precedence order: invalid_range / invalid_date / missing_value > type_inconsistency > spelling_typo > casing > whitespace > outlier
    precedence = {
        "invalid_range": 10,
        "invalid_date": 10,
        "missing_value": 9,
        "type_inconsistency": 8,
        "exact_duplicate": 7,
        "spelling_typo": 6,
        "casing_inconsistency": 5,
        "whitespace_inconsistency": 4,
        "near_duplicate": 3,
        "outlier": 2
    }

    cell_issue_map = {}
    row_wide_issues = []

    for issue in all_raw_issues:
        col = issue.get("column")
        row = issue.get("row")
        
        if col == "__all__":
            row_wide_issues.append(issue)
            continue

        key = (row, col)
        itype = issue.get("issue_type", "")
        current_prec = precedence.get(itype, 1)

        if key not in cell_issue_map:
            cell_issue_map[key] = issue
        else:
            existing = cell_issue_map[key]
            existing_prec = precedence.get(existing.get("issue_type", ""), 1)
            if current_prec > existing_prec:
                cell_issue_map[key] = issue

    deduplicated_issues = row_wide_issues + list(cell_issue_map.values())

    # Assign issue_id and final routing classification
    final_issues = []
    for idx, issue in enumerate(deduplicated_issues, 1):
        issue["issue_id"] = f"ISSUE_{idx:04d}"
        corr_conf = float(issue.get("correction_confidence", 0.0))
        
        # Decision engine routing:
        # High confidence (>= auto_thresh) and not explicitly marked human review
        if corr_conf >= auto_thresh and not issue.get("is_human_review_required", False):
            issue["routing_decision"] = "AUTO_CORRECT"
        elif 0.70 <= corr_conf < auto_thresh:
            issue["routing_decision"] = "SUGGEST_REVIEW"
        else:
            issue["routing_decision"] = "HUMAN_REVIEW_REQUIRED"
            
        final_issues.append(issue)

    logger.info(f"Total detected issues: {len(final_issues)} ({sum(1 for i in final_issues if i['routing_decision'] == 'AUTO_CORRECT')} auto-correctable)")
    return final_issues
