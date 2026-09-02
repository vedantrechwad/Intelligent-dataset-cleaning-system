from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from src.utils.helpers import logger

def evaluate_downstream_ml(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    target_column: str,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Rigorously trains identical ML models on Corrupted vs Cleaned datasets
    using identical seeds, train/test splits, and preprocessing.
    Evaluates classification or regression metrics.
    """
    if target_column not in dirty_df.columns or target_column not in clean_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in both datasets.")

    def preprocess_for_model(df_in: pd.DataFrame, target_col: str):
        # Keep numeric features and one-hot encode low-cardinality categoricals
        X = df_in.drop(columns=[target_col]).copy()
        y = df_in[target_col].copy()

        # Simple robust imputation for the ML baseline
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                med = X[col].median()
                X[col] = X[col].fillna(med if pd.notna(med) else 0)
            else:
                X[col] = X[col].fillna("Missing").astype(str)

        X = pd.get_dummies(X, drop_first=True)
        return X, y

    # Determine if task is classification or regression
    target_series = clean_df[target_column].dropna()
    is_classification = False
    if target_series.nunique() <= 10 or pd.api.types.is_object_dtype(target_series) or pd.api.types.is_bool_dtype(target_series):
        is_classification = True

    try:
        X_dirty, y_dirty = preprocess_for_model(dirty_df, target_column)
        X_clean, y_clean = preprocess_for_model(clean_df, target_column)

        # Align columns
        common_cols = list(set(X_dirty.columns).intersection(set(X_clean.columns)))
        if not common_cols:
            return {"error": "Insufficient overlapping features for ML model evaluation."}

        X_dirty = X_dirty[common_cols]
        X_clean = X_clean[common_cols]

        if is_classification:
            # Classification evaluation
            # Handle potential missing in target
            y_dirty = y_dirty.fillna(y_dirty.mode()[0] if not y_dirty.mode().empty else "Unknown").astype(str)
            y_clean = y_clean.fillna(y_clean.mode()[0] if not y_clean.mode().empty else "Unknown").astype(str)

            # Locked split
            X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(X_dirty, y_dirty, test_size=0.3, random_state=random_seed)
            X_c_tr, X_c_te, y_c_tr, y_c_te = train_test_split(X_clean, y_clean, test_size=0.3, random_state=random_seed)

            model_dirty = RandomForestClassifier(n_estimators=50, random_state=random_seed)
            model_dirty.fit(X_d_tr, y_d_tr)
            preds_dirty = model_dirty.predict(X_d_te)

            model_clean = RandomForestClassifier(n_estimators=50, random_state=random_seed)
            model_clean.fit(X_c_tr, y_c_tr)
            preds_clean = model_clean.predict(X_c_te)

            results = {
                "task_type": "Classification",
                "target_column": target_column,
                "model": "Random Forest Classifier",
                "corrupted_metrics": {
                    "accuracy": round(float(accuracy_score(y_d_te, preds_dirty)), 4),
                    "f1_weighted": round(float(f1_score(y_d_te, preds_dirty, average="weighted", zero_division=0)), 4),
                    "precision_weighted": round(float(precision_score(y_d_te, preds_dirty, average="weighted", zero_division=0)), 4),
                    "recall_weighted": round(float(recall_score(y_d_te, preds_dirty, average="weighted", zero_division=0)), 4)
                },
                "cleaned_metrics": {
                    "accuracy": round(float(accuracy_score(y_c_te, preds_clean)), 4),
                    "f1_weighted": round(float(f1_score(y_c_te, preds_clean, average="weighted", zero_division=0)), 4),
                    "precision_weighted": round(float(precision_score(y_c_te, preds_clean, average="weighted", zero_division=0)), 4),
                    "recall_weighted": round(float(recall_score(y_c_te, preds_clean, average="weighted", zero_division=0)), 4)
                },
                "disclaimer": "A cleaner dataset improves training stability, but does not guarantee improved test metrics on all distributions."
            }
        else:
            # Regression evaluation
            y_dirty = pd.to_numeric(y_dirty, errors="coerce").fillna(float(target_series.median()))
            y_clean = pd.to_numeric(y_clean, errors="coerce").fillna(float(target_series.median()))

            X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(X_dirty, y_dirty, test_size=0.3, random_state=random_seed)
            X_c_tr, X_c_te, y_c_tr, y_c_te = train_test_split(X_clean, y_clean, test_size=0.3, random_state=random_seed)

            model_dirty = RandomForestRegressor(n_estimators=50, random_state=random_seed)
            model_dirty.fit(X_d_tr, y_d_tr)
            preds_dirty = model_dirty.predict(X_d_te)

            model_clean = RandomForestRegressor(n_estimators=50, random_state=random_seed)
            model_clean.fit(X_c_tr, y_c_tr)
            preds_clean = model_clean.predict(X_c_te)

            results = {
                "task_type": "Regression",
                "target_column": target_column,
                "model": "Random Forest Regressor",
                "corrupted_metrics": {
                    "r2_score": round(float(r2_score(y_d_te, preds_dirty)), 4),
                    "mae": round(float(mean_absolute_error(y_d_te, preds_dirty)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_d_te, preds_dirty))), 4)
                },
                "cleaned_metrics": {
                    "r2_score": round(float(r2_score(y_c_te, preds_clean)), 4),
                    "mae": round(float(mean_absolute_error(y_c_te, preds_clean)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_c_te, preds_clean))), 4)
                },
                "disclaimer": "A cleaner dataset improves training stability, but does not guarantee improved test metrics on all distributions."
            }

        return results

    except Exception as e:
        logger.error(f"Error in downstream ML evaluation: {e}")
        return {"error": f"Downstream evaluation failed: {str(e)}"}
