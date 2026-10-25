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

        # Skip columns that are predominantly numeric (even if formatted as strings/objects)
        non_blanks = series.dropna().astype(str).str.strip()
        non_blanks = non_blanks[~non_blanks.isin(["", "nan", "none", "null"])]
        if len(non_blanks) > 0 and pd.to_numeric(non_blanks, errors="coerce").notna().mean() >= 0.70:
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

        # Pre-compute which unique values have whitespace or casing issues
        bad_val_map = {}
        for raw_val in exact_counts.keys():
            raw_str = str(raw_val)
            normalized_ws = " ".join(raw_str.strip().split())
            has_ws = (raw_str != normalized_ws)
            
            key = normalized_ws.lower()
            canonical, dom_count = canonical_counts.get(key, (normalized_ws, 1))
            has_case = (normalized_ws != canonical and dom_count > 1) if has_ws else (raw_str != canonical and dom_count > 1)
            
            if has_ws or has_case:
                bad_val_map[raw_val] = (has_ws, normalized_ws, has_case, canonical)

        # Only iterate over rows if there are detected unique inconsistencies
        if bad_val_map:
            bad_keys = set(bad_val_map.keys())
            for row_idx, val in series.items():
                if val in bad_keys:
                    has_ws, normalized_ws, has_case, canonical = bad_val_map[val]
                    raw_str = str(val)

                    if has_ws:
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

                    if has_case:
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
        is_code_col = any(w in col.lower() for w in ["state", "country", "code", "abbr", "currency", "iso", "zip"]) or (len(unique_categories) > 3 and all(len(c) <= 3 for c in unique_categories[:10]))

        if len(unique_categories) > 1 and len(unique_categories) <= 100 and not is_code_col:
            # Sort categories by total occurrences descending
            sorted_cats = sorted(
                unique_categories, 
                key=lambda k: canonical_counts[k][1], 
                reverse=True
            )
            
            # Dominant categories with >= 3 occurrences or > 10% of data
            dominant_cats = [c for c in sorted_cats if canonical_counts[c][1] >= 2]
            
            for rare_cat in sorted_cats:
                # Never treat 1 or 2 letter codes as typos of other words
                if len(rare_cat) <= 2:
                    continue

                rare_count = canonical_counts[rare_cat][1]
                
                candidates = []
                for dom_cat in dominant_cats:
                    if rare_cat == dom_cat or len(dom_cat) <= 2:
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
                    elif 3 <= len(dom_cat) <= 5 and 3 <= len(rare_cat) <= 5:
                        if set(rare_cat) == set(dom_cat) and len(rare_cat) == len(dom_cat):
                            is_obvious_typo = True # Exact transposition like yse -> yes
                        elif len(dom_cat) >= 4 and len(rare_cat) >= 4:
                            match_chars = sum(1 for c in rare_cat if c in dom_cat)
                            if match_chars >= len(dom_cat) - 1 and abs(len(rare_cat) - len(dom_cat)) <= 1 and sim >= 75.0:
                                is_obvious_typo = True # Single typo in >=4 char word (e.g. n0 -> no disabled, but tehl -> tell allowed)

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
