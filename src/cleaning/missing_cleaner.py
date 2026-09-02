from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from src.utils.helpers import load_config, logger

def clean_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    custom_constants: Dict[str, Any] = None,
    impute_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Imputes missing values using selected strategy:
      - Numerical: 'median', 'mean', 'knn', 'constant', or 'skip'
      - Categorical: 'mode', 'constant', or 'skip'
    Tracks and returns comprehensive modification log for each imputed cell.
    """
    cleaned_df = df.copy(deep=True)
    logs = []
    custom_constants = custom_constants or {}
    columns_to_process = impute_columns if impute_columns is not None else list(cleaned_df.columns)

    # Pre-clean string representations of missing values (e.g. 'N/A', 'null', whitespace)
    cfg = load_config()
    missing_tokens = {str(m).strip().lower() for m in cfg.get("missing_representations", ["nan", "null", "n/a", "unknown"])}
    
    for col in columns_to_process:
        if col not in cleaned_df.columns:
            continue
        series = cleaned_df[col]
        # Replace empty strings and missing tokens with np.nan
        cleaned_df[col] = series.apply(
            lambda v: np.nan if pd.isna(v) or (isinstance(v, str) and (not v.strip() or v.strip().lower() in missing_tokens)) else v
        )

    # 1. Numerical Imputation
    numeric_cols = [c for c in columns_to_process if pd.api.types.is_numeric_dtype(cleaned_df[c])]
    
    if numeric_strategy == "knn" and len(numeric_cols) >= 2:
        try:
            imputer = KNNImputer(n_neighbors=5)
            imputed_arr = imputer.fit_transform(cleaned_df[numeric_cols])
            imputed_sub = pd.DataFrame(imputed_arr, columns=numeric_cols, index=cleaned_df.index)
            
            for col in numeric_cols:
                missing_mask = cleaned_df[col].isna()
                for row_idx in cleaned_df[missing_mask].index:
                    imp_val = round(float(imputed_sub.at[row_idx, col]), 3)
                    cleaned_df.at[row_idx, col] = imp_val
                    logs.append({
                        "row": int(row_idx),
                        "column": col,
                        "original_value": None,
                        "issue_type": "missing_value",
                        "action": "imputed_knn",
                        "corrected_value": imp_val,
                        "confidence": 0.90,
                        "reason": f"Imputed via KNNImputer (k=5)"
                    })
        except Exception as e:
            logger.warning(f"KNN imputation fallback to median: {e}")
            numeric_strategy = "median"

    if numeric_strategy in ["median", "mean", "constant"]:
        for col in numeric_cols:
            missing_mask = cleaned_df[col].isna()
            if not missing_mask.any():
                continue

            valid_vals = cleaned_df[col].dropna()
            if len(valid_vals) == 0:
                continue

            if numeric_strategy == "median":
                fill_val = round(float(valid_vals.median()), 3)
            elif numeric_strategy == "mean":
                fill_val = round(float(valid_vals.mean()), 3)
            else:
                fill_val = custom_constants.get(col, 0.0)

            for row_idx in cleaned_df[missing_mask].index:
                cleaned_df.at[row_idx, col] = fill_val
                logs.append({
                    "row": int(row_idx),
                    "column": col,
                    "original_value": None,
                    "issue_type": "missing_value",
                    "action": f"imputed_{numeric_strategy}",
                    "corrected_value": fill_val,
                    "confidence": 0.88,
                    "reason": f"Imputed with column {numeric_strategy} value ({fill_val})"
                })

    # 2. Categorical Imputation
    cat_cols = [c for c in columns_to_process if c not in numeric_cols]
    for col in cat_cols:
        missing_mask = cleaned_df[col].isna()
        if not missing_mask.any():
            continue

        valid_vals = cleaned_df[col].dropna().astype(str).str.strip()
        if len(valid_vals) == 0:
            fill_val = "Unknown"
        elif categorical_strategy == "mode":
            mode_vals = valid_vals.mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
        else:
            fill_val = custom_constants.get(col, "Unknown")

        for row_idx in cleaned_df[missing_mask].index:
            cleaned_df.at[row_idx, col] = fill_val
            logs.append({
                "row": int(row_idx),
                "column": col,
                "original_value": None,
                "issue_type": "missing_value",
                "action": f"imputed_{categorical_strategy}",
                "corrected_value": fill_val,
                "confidence": 0.85,
                "reason": f"Imputed with categorical {categorical_strategy} value ('{fill_val}')"
            })

    return cleaned_df, logs
