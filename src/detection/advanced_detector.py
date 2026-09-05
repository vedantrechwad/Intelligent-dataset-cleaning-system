import pandas as pd
import numpy as np
from typing import List, Dict, Any

def detect_advanced_anomalies(df: pd.DataFrame, target_variable: str = None) -> List[Dict[str, Any]]:
    """
    Detects complex anomalies including:
    - Multivariate anomalies via Isolation Forest
    - High-cardinality and constant columns
    - Class imbalance and rare categories
    - Highly correlated columns (data leakage)
    - Basic cross-column contradictions
    """
    issues = []

    # 1. Constant Columns and High Cardinality
    for col in df.columns:
        valid_series = df[col].dropna()
        if len(valid_series) == 0:
            continue
        
        nunique = valid_series.nunique()
        total = len(valid_series)
        
        if nunique == 1:
            issues.append({
                "row": None,
                "column": col,
                "original_value": None,
                "issue_type": "constant_column",
                "detection_method": ["advanced_detector"],
                "detection_confidence": 1.0,
                "correction_confidence": 0.0,
                "suggested_action": "report_only",
                "suggested_value": None,
                "reason": f"Column '{col}' has a single constant value. It provides no variance.",
                "is_human_review_required": False
            })
        elif pd.api.types.is_object_dtype(valid_series) and nunique > 0:
            # High cardinality: if > 90% unique and it's a string, likely an ID or free text
            if (nunique / total) > 0.90 and total > 50:
                issues.append({
                    "row": None,
                    "column": col,
                    "original_value": None,
                    "issue_type": "high_cardinality",
                    "detection_method": ["advanced_detector"],
                    "detection_confidence": 1.0,
                    "correction_confidence": 0.0,
                    "suggested_action": "report_only",
                    "suggested_value": None,
                    "reason": f"Column '{col}' is categorical but has {(nunique/total)*100:.1f}% unique values (likely an ID or free-text).",
                    "is_human_review_required": False
                })

    # 2. Class Imbalance and Rare Categories
    for col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
        valid_series = df[col].dropna()
        if len(valid_series) == 0:
            continue
        
        value_counts = valid_series.value_counts(normalize=True)
        
        # Class imbalance (top class > 95%)
        if value_counts.iloc[0] > 0.95 and len(value_counts) > 1:
            issues.append({
                "row": None,
                "column": col,
                "original_value": None,
                "issue_type": "class_imbalance",
                "detection_method": ["advanced_detector"],
                "detection_confidence": 1.0,
                "correction_confidence": 0.0,
                "suggested_action": "report_only",
                "suggested_value": None,
                "reason": f"Column '{col}' is heavily imbalanced. Top class '{value_counts.index[0]}' constitutes {value_counts.iloc[0]*100:.1f}% of data.",
                "is_human_review_required": False
            })
            
        # Rare categories (< 1%)
        for cat, freq in value_counts.items():
            if freq < 0.01 and len(valid_series) > 100:
                # Find rows with this rare category
                rare_rows = df[df[col] == cat].index
                for r in rare_rows:
                    issues.append({
                        "row": int(r),
                        "column": col,
                        "original_value": cat,
                        "issue_type": "rare_category",
                        "detection_method": ["advanced_detector"],
                        "detection_confidence": 0.9,
                        "correction_confidence": 0.0,
                        "suggested_action": "report_only",
                        "suggested_value": None,
                        "reason": f"Category '{cat}' is extremely rare ({freq*100:.2f}% frequency).",
                        "is_human_review_required": False
                    })

    # 3. Highly Correlated / Data Leakage
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        for col1 in upper_tri.columns:
            for col2 in upper_tri.index:
                val = upper_tri.loc[col2, col1]
                if pd.notna(val) and val > 0.98:
                    issue_type = "data_leakage" if target_variable in [col1, col2] else "highly_correlated"
                    msg = f"Columns '{col1}' and '{col2}' are nearly perfectly correlated (r={val:.3f})."
                    if issue_type == "data_leakage":
                        msg += " Potential DATA LEAKAGE against target variable!"
                    issues.append({
                        "row": None,
                        "column": col1,
                        "original_value": None,
                        "issue_type": issue_type,
                        "detection_method": ["advanced_detector"],
                        "detection_confidence": 1.0,
                        "correction_confidence": 0.0,
                        "suggested_action": "report_only",
                        "suggested_value": None,
                        "reason": msg,
                        "is_human_review_required": False
                    })

    # 4. Multivariate Anomalies (Isolation Forest)
    if numeric_df.shape[1] >= 2 and len(numeric_df) > 50:
        try:
            from sklearn.ensemble import IsolationForest
            clean_num = numeric_df.dropna()
            if len(clean_num) > 50:
                iso = IsolationForest(contamination=0.01, random_state=42)
                preds = iso.fit_predict(clean_num)
                anomaly_indices = clean_num.index[preds == -1]
                for r in anomaly_indices:
                    issues.append({
                        "row": int(r),
                        "column": "__all__",
                        "original_value": None,
                        "issue_type": "multivariate_anomaly",
                        "detection_method": ["isolation_forest"],
                        "detection_confidence": 0.85,
                        "correction_confidence": 0.0,
                        "suggested_action": "report_only",
                        "suggested_value": None,
                        "reason": f"Row {r} is a multivariate statistical anomaly across numeric columns.",
                        "is_human_review_required": False
                    })
        except ImportError:
            pass

    # 5. Cross-Column Contradictions (Heuristics)
    # E.g., Age < 18 and Marital Status == Married
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    marital_cols = [c for c in df.columns if 'marital' in c.lower() or 'marriage' in c.lower()]
    if age_cols and marital_cols:
        age_col = age_cols[0]
        mar_col = marital_cols[0]
        
        if pd.api.types.is_numeric_dtype(df[age_col]):
            mask = (df[age_col] < 16) & (df[mar_col].astype(str).str.lower().str.contains('marri', na=False))
            for r in df[mask].index:
                issues.append({
                    "row": int(r),
                    "column": age_col,
                    "original_value": df.loc[r, age_col],
                    "issue_type": "cross_column_contradiction",
                    "detection_method": ["advanced_detector"],
                    "detection_confidence": 0.95,
                    "correction_confidence": 0.5,
                    "suggested_action": "flag_for_human_review",
                    "suggested_value": None,
                    "reason": f"Logical contradiction: Age is {df.loc[r, age_col]} but Marital Status is '{df.loc[r, mar_col]}'.",
                    "is_human_review_required": True
                })

    return issues
