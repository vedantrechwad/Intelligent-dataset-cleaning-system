from typing import List, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from src.utils.helpers import load_config, logger

def detect_numerical_outliers(
    df: pd.DataFrame, 
    iqr_multiplier: float = 1.5, 
    zscore_threshold: float = 3.0,
    contamination: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Detects numerical outliers using an ensemble of:
      1. IQR Method
      2. Z-Score Method
      3. Isolation Forest (Multivariate / Feature-level)
    Combines normalized signals. Outliers are strictly flagged and explained;
    they are NEVER automatically removed by default.
    """
    cfg = load_config()
    thresh_cfg = cfg.get("thresholds", {})
    iqr_multiplier = thresh_cfg.get("outlier_iqr_multiplier", iqr_multiplier)
    zscore_threshold = thresh_cfg.get("outlier_zscore_threshold", zscore_threshold)
    contamination = thresh_cfg.get("isolation_forest_contamination", contamination)

    outlier_issues = []
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    if not numeric_cols:
        return outlier_issues

    # 1. Isolation Forest on available numeric features
    iso_forest_scores = {}
    clean_numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(df[numeric_cols].median(numeric_only=True))
    
    if len(clean_numeric_df) >= 15 and len(numeric_cols) >= 1:
        try:
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(clean_numeric_df)
            decision_scores = iso.decision_function(clean_numeric_df)
            # Normalize decision score to [0, 1] anomaly likelihood
            min_score, max_score = decision_scores.min(), decision_scores.max()
            if max_score > min_score:
                norm_iso = 1.0 - ((decision_scores - min_score) / (max_score - min_score))
            else:
                norm_iso = np.zeros(len(decision_scores))
            
            for idx, (p, score) in enumerate(zip(clean_numeric_df.index, norm_iso)):
                iso_forest_scores[idx] = (p == -1, float(score))
        except Exception as e:
            logger.warning(f"Isolation Forest skipped: {e}")

    # 2. Per-column IQR & Z-score evaluation
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid_series = series.dropna()

        if len(valid_series) < 5:
            continue

        # IQR
        q1 = float(np.percentile(valid_series, 25))
        q3 = float(np.percentile(valid_series, 75))
        iqr = q3 - q1
        lower_iqr = q1 - (iqr_multiplier * iqr)
        upper_iqr = q3 + (iqr_multiplier * iqr)

        # Z-scores
        std_dev = valid_series.std()
        mean_val = valid_series.mean()
        z_scores = {}
        if std_dev > 1e-6:
            z_scores = ((valid_series - mean_val).abs() / std_dev).to_dict()

        for row_idx, val in valid_series.items():
            signals = []
            score_weights = []

            # IQR signal
            is_iqr = (val < lower_iqr) or (val > upper_iqr)
            if is_iqr:
                signals.append("IQR")
                dist = abs(val - (upper_iqr if val > upper_iqr else lower_iqr))
                score_weights.append(min(1.0, dist / (iqr + 1e-6)))

            # Z-score signal
            z = z_scores.get(row_idx, 0.0)
            is_zscore = z > zscore_threshold
            if is_zscore:
                signals.append(f"Z-score (|z|={z:.2f})")
                score_weights.append(min(1.0, z / (zscore_threshold * 2)))

            # Isolation Forest signal
            is_iso, iso_anom = iso_forest_scores.get(row_idx, (False, 0.0))
            if is_iso:
                signals.append("Isolation Forest")
                score_weights.append(iso_anom)

            # If at least 2 methods or extreme single method flagged it
            if (len(signals) >= 2) or (is_zscore and z > 4.0) or (is_iqr and abs(val - mean_val) > 3 * iqr):
                composite_conf = float(np.mean(score_weights)) if score_weights else 0.70
                composite_conf = max(0.50, min(0.99, composite_conf))

                outlier_issues.append({
                    "row": int(row_idx),
                    "column": col,
                    "original_value": float(val),
                    "issue_type": "outlier",
                    "detection_method": ["outlier_ensemble"] + signals,
                    "detection_confidence": round(composite_conf, 3),
                    "correction_confidence": 0.30, # Low correction confidence: don't alter without user choice
                    "suggested_action": "flag_only", # Configurable user choice
                    "suggested_value": round(float(valid_series.median()), 2),
                    "reason": f"Value {val} flagged by {', '.join(signals)} (Q1={q1:.1f}, Q3={q3:.1f})",
                    "is_human_review_required": True
                })

    return outlier_issues
