"""
src/evaluation/downstream_evaluation.py
Empirical Downstream Machine Learning Utility Benchmarking Engine.

Rigorously benchmarks predictive accuracy, generalization, and stability
of Machine Learning models trained on Corrupted / Raw Data vs. Cleaned Data.
Follows the experimental design of CleanML (Li et al., IEEE TKDE 2021).
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from src.utils.helpers import logger


COMMON_TARGET_NAMES = [
    "churn", "target", "label", "class", "diagnosis", "income",
    "survived", "price", "rating", "type", "room_type", "poutcome",
    "deposit", "y", "default", "g3", "status", "category"
]


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """Automatically infers candidate ML target column based on naming and entropy."""
    cols_lower = {c.lower().strip().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    
    # Priority 1: Check known candidate names
    for candidate in COMMON_TARGET_NAMES:
        if candidate in cols_lower:
            return cols_lower[candidate]

    # Priority 2: Look for binary or low-cardinality categorical column at the end of the table
    for col in reversed(df.columns):
        s = df[col].dropna()
        n_uniq = s.nunique()
        if 2 <= n_uniq <= 10 and len(s) >= 20:
            return col

    # Priority 3: Last numeric column
    for col in reversed(df.columns):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) >= 20 and s.nunique() > 5:
            return col

    return None


def evaluate_downstream_ml(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    target_column: Optional[str] = None,
    random_seed: int = 42,
    max_eval_rows: int = 25000
) -> Dict[str, Any]:
    """
    Rigorously trains identical ML models on Dirty vs Cleaned datasets
    using identical seeds, train/test splits, and preprocessing.
    Computes performance metrics, absolute deltas, and percentage gains.
    """
    if target_column is None:
        target_column = infer_target_column(clean_df) or infer_target_column(dirty_df)

    if not target_column or target_column not in dirty_df.columns or target_column not in clean_df.columns:
        return {"error": f"Target column '{target_column}' not found or could not be inferred."}

    # Downsample if matrix is massive for fast interactive evaluation
    n_rows = min(len(dirty_df), len(clean_df))
    if n_rows > max_eval_rows:
        dirty_sub = dirty_df.iloc[:max_eval_rows].copy()
        clean_sub = clean_df.iloc[:max_eval_rows].copy()
    else:
        dirty_sub = dirty_df.copy()
        clean_sub = clean_df.copy()

    def preprocess_features(df_in: pd.DataFrame, target_col: str):
        X = df_in.drop(columns=[target_col], errors="ignore").copy()
        y = df_in[target_col].copy()

        # Handle numeric features
        for col in X.columns:
            num_series = pd.to_numeric(X[col], errors="coerce")
            if num_series.notna().mean() >= 0.75:
                med = num_series.median()
                X[col] = num_series.fillna(med if pd.notna(med) else 0.0)
            else:
                X[col] = X[col].fillna("Missing").astype(str).str.strip().str.lower()
                # Cap cardinality
                top_cats = X[col].value_counts().head(20).index
                X[col] = X[col].where(X[col].isin(top_cats), "Other")

        X = pd.get_dummies(X, drop_first=True)
        return X, y

    # Determine if task is classification or regression based on numeric parsability and cardinality
    target_series = clean_sub[target_column].replace("", np.nan).dropna()
    num_target = pd.to_numeric(
        target_series.astype(str).str.replace(r"[$,€£¥]", "", regex=True).str.replace(",", "").str.strip(),
        errors="coerce"
    )
    is_numeric_target = (num_target.notna().mean() >= 0.80) and (num_target.nunique() > 10)
    is_classification = not is_numeric_target

    try:
        X_dirty, y_dirty = preprocess_features(dirty_sub, target_column)
        X_clean, y_clean = preprocess_features(clean_sub, target_column)

        # Align columns
        all_cols = sorted(list(set(X_dirty.columns).union(set(X_clean.columns))))
        X_dirty = X_dirty.reindex(columns=all_cols, fill_value=0)
        X_clean = X_clean.reindex(columns=all_cols, fill_value=0)

        # Robust train/test split aligned by index intersection (handles dropped duplicates/rows)
        common_indices = clean_sub.index.intersection(dirty_sub.index)
        if len(common_indices) < 10:
            # Fallback to positional indices
            n_min = min(len(X_dirty), len(X_clean))
            common_idx_pos = np.arange(n_min)
            train_pos, test_pos = train_test_split(common_idx_pos, test_size=0.3, random_state=random_seed)
            X_d_tr, X_d_te = X_dirty.iloc[train_pos], X_dirty.iloc[test_pos]
            y_d_tr, y_d_te = y_dirty.iloc[train_pos], y_dirty.iloc[test_pos]
            X_c_tr, X_c_te = X_clean.iloc[train_pos], X_clean.iloc[test_pos]
            y_c_tr, y_c_te = y_clean.iloc[train_pos], y_clean.iloc[test_pos]
        else:
            train_idx, test_idx = train_test_split(common_indices, test_size=0.3, random_state=random_seed)
            X_d_tr, X_d_te = X_dirty.loc[train_idx], X_dirty.loc[test_idx]
            y_d_tr, y_d_te = y_dirty.loc[train_idx], y_dirty.loc[test_idx]
            X_c_tr, X_c_te = X_clean.loc[train_idx], X_clean.loc[test_idx]
            y_c_tr, y_c_te = y_clean.loc[train_idx], y_clean.loc[test_idx]

        if is_classification:
            y_d_tr = y_d_tr.fillna("Unknown").astype(str)
            y_d_te = y_d_te.fillna("Unknown").astype(str)
            y_c_tr = y_c_tr.fillna("Unknown").astype(str)
            y_c_te = y_c_te.fillna("Unknown").astype(str)

            # Model: Random Forest Classifier
            rf_dirty = RandomForestClassifier(n_estimators=40, max_depth=12, random_state=random_seed, n_jobs=-1)
            rf_dirty.fit(X_d_tr, y_d_tr)
            # Evaluate both models on the clean test ground truth (CleanML methodology)
            preds_d = rf_dirty.predict(X_c_te)

            rf_clean = RandomForestClassifier(n_estimators=40, max_depth=12, random_state=random_seed, n_jobs=-1)
            rf_clean.fit(X_c_tr, y_c_tr)
            preds_c = rf_clean.predict(X_c_te)

            acc_d = float(accuracy_score(y_c_te, preds_d))
            acc_c = float(accuracy_score(y_c_te, preds_c))

            f1_d = float(f1_score(y_c_te, preds_d, average="weighted", zero_division=0))
            f1_c = float(f1_score(y_c_te, preds_c, average="weighted", zero_division=0))

            prec_d = float(precision_score(y_c_te, preds_d, average="weighted", zero_division=0))
            prec_c = float(precision_score(y_c_te, preds_c, average="weighted", zero_division=0))

            rec_d = float(recall_score(y_c_te, preds_d, average="weighted", zero_division=0))
            rec_c = float(recall_score(y_c_te, preds_c, average="weighted", zero_division=0))

            delta_acc = acc_c - acc_d
            delta_f1 = f1_c - f1_d
            pct_gain_f1 = ((f1_c - f1_d) / max(f1_d, 1e-6)) * 100.0

            return {
                "status": "success",
                "task_type": "Classification",
                "target_column": target_column,
                "n_samples_evaluated": len(X_clean),
                "model_name": "Random Forest Classifier (40 trees, depth=12)",
                "dirty_metrics": {
                    "accuracy": round(acc_d, 4),
                    "f1_score": round(f1_d, 4),
                    "precision": round(prec_d, 4),
                    "recall": round(rec_d, 4)
                },
                "corrupted_metrics": {
                    "accuracy": round(acc_d, 4),
                    "f1_score": round(f1_d, 4),
                    "precision": round(prec_d, 4),
                    "recall": round(rec_d, 4)
                },
                "cleaned_metrics": {
                    "accuracy": round(acc_c, 4),
                    "f1_score": round(f1_c, 4),
                    "precision": round(prec_c, 4),
                    "recall": round(rec_c, 4)
                },
                "deltas": {
                    "delta_accuracy": round(delta_acc, 4),
                    "delta_f1": round(delta_f1, 4),
                    "pct_gain_f1": round(pct_gain_f1, 2)
                },
                "metrics_delta": {
                    "delta_accuracy": round(delta_acc, 4),
                    "delta_f1": round(delta_f1, 4),
                    "f1_lift_percent": round(pct_gain_f1, 2)
                },
                "improved": delta_f1 >= 0.0 or delta_acc >= 0.0
            }

        else:
            # Regression task
            def clean_num_target(s: pd.Series) -> pd.Series:
                s_str = s.astype(str).str.replace(r"[$,€£¥]", "", regex=True).str.replace(",", "").str.strip()
                return pd.to_numeric(s_str, errors="coerce").fillna(0.0)

            y_d_tr_num = clean_num_target(y_d_tr)
            y_d_te_num = clean_num_target(y_d_te)
            y_c_tr_num = clean_num_target(y_c_tr)
            y_c_te_num = clean_num_target(y_c_te)

            rf_dirty = RandomForestRegressor(n_estimators=40, max_depth=12, random_state=random_seed, n_jobs=-1)
            rf_dirty.fit(X_d_tr, y_d_tr_num)
            preds_d = rf_dirty.predict(X_c_te)

            rf_clean = RandomForestRegressor(n_estimators=40, max_depth=12, random_state=random_seed, n_jobs=-1)
            rf_clean.fit(X_c_tr, y_c_tr_num)
            preds_c = rf_clean.predict(X_c_te)

            r2_d = float(r2_score(y_c_te_num, preds_d))
            r2_c = float(r2_score(y_c_te_num, preds_c))

            mae_d = float(mean_absolute_error(y_c_te_num, preds_d))
            mae_c = float(mean_absolute_error(y_c_te_num, preds_c))

            rmse_d = float(np.sqrt(mean_squared_error(y_c_te_num, preds_d)))
            rmse_c = float(np.sqrt(mean_squared_error(y_c_te_num, preds_c)))

            delta_r2 = r2_c - r2_d
            delta_rmse = rmse_d - rmse_c  # Positive means error decreased

            return {
                "status": "success",
                "task_type": "Regression",
                "target_column": target_column,
                "n_samples_evaluated": len(X_clean),
                "model_name": "Random Forest Regressor (40 trees, depth=12)",
                "dirty_metrics": {
                    "r2_score": round(r2_d, 4),
                    "mae": round(mae_d, 4),
                    "rmse": round(rmse_d, 4)
                },
                "corrupted_metrics": {
                    "r2_score": round(r2_d, 4),
                    "mae": round(mae_d, 4),
                    "rmse": round(rmse_d, 4)
                },
                "cleaned_metrics": {
                    "r2_score": round(r2_c, 4),
                    "mae": round(mae_c, 4),
                    "rmse": round(rmse_c, 4)
                },
                "deltas": {
                    "delta_r2": round(delta_r2, 4),
                    "delta_rmse_reduction": round(delta_rmse, 4)
                },
                "metrics_delta": {
                    "delta_r2": round(delta_r2, 4),
                    "delta_rmse_reduction": round(delta_rmse, 4)
                },
                "improved": delta_r2 >= 0.0 or delta_rmse >= 0.0
            }

    except Exception as e:
        logger.error(f"Downstream ML evaluation failed: {e}")
        return {"error": f"Evaluation error: {str(e)}"}
