import re
from typing import List, Dict, Any, Tuple
import pandas as pd
from rapidfuzz import fuzz, process
from src.utils.helpers import load_config, logger

def detect_categorical_inconsistencies(
    df: pd.DataFrame,
    similarity_threshold: float = None
) -> List[Dict[str, Any]]:
    """
    Detects categorical inconsistencies:
      1. Whitespace padding and internal multiple spaces
      2. Casing variations (e.g., 'mumbai', 'MUMBAI', 'Mumbai')
      3. Spelling typos & minor edits via RapidFuzz string similarity (e.g., 'Mumabi' -> 'Mumbai')
    """
    cfg = load_config()
    if similarity_threshold is None:
        similarity_threshold = cfg.get("thresholds", {}).get("category_similarity_threshold", 82.0)

    inconsistency_issues = []

    for col in df.columns:
        series = df[col]
        # Skip purely numeric or datetime columns
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        valid_series = series.dropna().astype(str)
        if len(valid_series) == 0:
            continue

        # 1. Whitespace and Casing Inconsistencies
        # Group by stripped lowercase to find standard representative
        lower_to_canonical = {}
        canonical_counts = {}

        # First pass: count frequencies of exact representations
        exact_counts = valid_series.value_counts().to_dict()

        # Build dominant canonical form for each lowercase version
        for raw_val, count in exact_counts.items():
            cleaned_norm = " ".join(raw_val.strip().split())
            key = cleaned_norm.lower()
            if key not in canonical_counts or count > canonical_counts[key][1]:
                # Prefer Title Case or most frequent
                canonical_counts[key] = (cleaned_norm.title() if cleaned_norm.islower() else cleaned_norm, count)

        for row_idx, val in series.items():
            if pd.isna(val) or val is None:
                continue

            raw_str = str(val)
            normalized_ws = " ".join(raw_str.strip().split())

            # Whitespace issue
            if raw_str != normalized_ws:
                inconsistency_issues.append({
                    "row": int(row_idx),
                    "column": col,
                    "original_value": raw_str,
                    "issue_type": "whitespace_inconsistency",
                    "detection_method": ["whitespace_analyzer"],
                    "detection_confidence": 1.0,
                    "correction_confidence": 1.0,
                    "suggested_action": "normalize_whitespace",
                    "suggested_value": normalized_ws,
                    "reason": f"Value '{raw_str}' contains irregular leading/trailing or multiple whitespaces",
                    "is_human_review_required": False
                })
                raw_str = normalized_ws

            # Casing issue
            key = normalized_ws.lower()
            canonical, dom_count = canonical_counts.get(key, (normalized_ws, 1))
            if raw_str != canonical and dom_count > 1:
                inconsistency_issues.append({
                    "row": int(row_idx),
                    "column": col,
                    "original_value": raw_str,
                    "issue_type": "casing_inconsistency",
                    "detection_method": ["casing_analyzer"],
                    "detection_confidence": 1.0,
                    "correction_confidence": 0.98,
                    "suggested_action": "normalize_casing",
                    "suggested_value": canonical,
                    "reason": f"Value '{raw_str}' has inconsistent casing compared to canonical '{canonical}'",
                    "is_human_review_required": False
                })

        # 2. Spelling Typos & Fuzzy Variations across distinct categories
        unique_categories = list(set(canonical_counts.keys()))
        if len(unique_categories) > 1 and len(unique_categories) <= 100:
            # Sort categories by total occurrences descending
            sorted_cats = sorted(
                unique_categories, 
                key=lambda k: canonical_counts[k][1], 
                reverse=True
            )
            
            # Dominant categories with >= 3 occurrences or > 10% of data
            dominant_cats = [c for c in sorted_cats if canonical_counts[c][1] >= 2]
            
            for rare_cat in sorted_cats:
                rare_count = canonical_counts[rare_cat][1]
                
                candidates = []
                for dom_cat in dominant_cats:
                    if rare_cat == dom_cat:
                        continue
                        
                    dom_count = canonical_counts[dom_cat][1]
                    if dom_count <= rare_count:
                        continue

                    # Strict semantic boundary check (e.g. 'No' vs 'No internet service')
                    if abs(len(rare_cat) - len(dom_cat)) > 3:
                        continue

                    sim = fuzz.ratio(rare_cat, dom_cat)
                    
                    is_obvious_typo = False
                    if sim >= similarity_threshold:
                        is_obvious_typo = True
                    elif len(dom_cat) <= 4 and len(rare_cat) <= 4:
                        if set(rare_cat) == set(dom_cat) and len(rare_cat) == len(dom_cat):
                            is_obvious_typo = True # Anagram/transposition like yse -> yes
                        else:
                            match_chars = sum(1 for c in rare_cat if c in dom_cat)
                            if match_chars >= len(dom_cat) - 1 and abs(len(rare_cat) - len(dom_cat)) <= 1:
                                is_obvious_typo = True # n0 -> no

                    if is_obvious_typo:
                        candidates.append((dom_cat, dom_count, sim))

                if not candidates:
                    continue

                is_ambiguous = len(candidates) > 1
                best_dom_cat, best_dom_count, best_sim = max(candidates, key=lambda x: (x[2], x[1]))
                
                freq_ratio = min(1.0, (best_dom_count - rare_count) / max(1, best_dom_count))
                score = (0.50 * (best_sim / 100.0)) + (0.35 * freq_ratio) + 0.15
                
                if not is_ambiguous:
                    score = max(score, 0.96) # Boost to ensure auto-correction for unambiguous typos
                else:
                    score = min(score, 0.90) # Downgrade to require human review

                score = round(min(0.99, score), 3)
                canonical_target = canonical_counts[best_dom_cat][0]
                
                for row_idx, val in series.items():
                    if pd.isna(val) or val is None:
                        continue
                    if str(val).strip().lower() == rare_cat:
                        inconsistency_issues.append({
                            "row": int(row_idx),
                            "column": col,
                            "original_value": str(val),
                            "issue_type": "spelling_typo",
                            "detection_method": ["rapidfuzz_similarity", "frequency_analysis"],
                            "detection_confidence": score,
                            "correction_confidence": score,
                            "suggested_action": "auto_correct" if score >= 0.95 else "flag_for_review",
                            "suggested_value": canonical_target,
                            "reason": f"Value '{val}' matched to '{canonical_target}' (sim: {best_sim:.0f}%, ambiguous: {is_ambiguous})",
                            "is_human_review_required": score < 0.95
                        })

    return inconsistency_issues
