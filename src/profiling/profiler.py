from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.ingestion.schema_detector import detect_dataset_schema
from src.utils.helpers import logger

def profile_dataset(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes a comprehensive profile for a tabular dataset.
    Calculates detailed metrics for numerical and categorical columns,
    as well as overall dataset-level statistics.
    """
    if df.empty:
        return {
            "row_count": 0,
            "column_count": 0,
            "total_cells": 0,
            "missing_cells": 0,
            "missing_percentage": 0.0,
            "duplicate_rows": 0,
            "columns": {}
        }

    if schema is None:
        schema = detect_dataset_schema(df)

    row_count = len(df)
    col_count = len(df.columns)
    total_cells = row_count * col_count
    
    # Dataset level missing & duplicates
    missing_cells = int(df.isna().sum().sum())
    missing_percentage = round((missing_cells / total_cells) * 100.0, 2) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = []
    categorical_cols = []
    date_cols = []
    column_profiles = {}

    for col in df.columns:
        col_series = df[col]
        col_schema = schema.get(col, {})
        logical_type = col_schema.get("logical_type", "unknown")
        
        col_missing = int(col_series.isna().sum())
        col_missing_pct = round((col_missing / row_count) * 100.0, 2) if row_count > 0 else 0.0
        unique_count = int(col_series.nunique(dropna=True))

        base_profile = {
            "column": col,
            "detected_type": logical_type,
            "semantic_hint": col_schema.get("semantic_hint", "none"),
            "missing_count": col_missing,
            "missing_percentage": col_missing_pct,
            "unique_count": unique_count,
            "is_nullable": col_missing > 0
        }

        # Numerical statistics
        if pd.api.types.is_numeric_dtype(col_series) or logical_type == "numeric":
            numeric_cols.append(col)
            # Safe numeric conversion for stats
            num_data = pd.to_numeric(col_series, errors="coerce").dropna()
            
            if len(num_data) > 0:
                q1 = float(np.percentile(num_data, 25))
                q3 = float(np.percentile(num_data, 75))
                iqr = float(q3 - q1)
                
                base_profile.update({
                    "min": float(num_data.min()),
                    "max": float(num_data.max()),
                    "mean": round(float(num_data.mean()), 3),
                    "median": round(float(num_data.median()), 3),
                    "std": round(float(num_data.std()), 3) if len(num_data) > 1 else 0.0,
                    "q1": round(q1, 3),
                    "q3": round(q3, 3),
                    "iqr": round(iqr, 3),
                    "zero_count": int((num_data == 0).sum()),
                    "negative_count": int((num_data < 0).sum())
                })
            else:
                base_profile.update({
                    "min": None, "max": None, "mean": None, "median": None,
                    "std": None, "q1": None, "q3": None, "iqr": None,
                    "zero_count": 0, "negative_count": 0
                })
                
        # Datetime statistics
        elif logical_type == "datetime" or pd.api.types.is_datetime64_any_dtype(col_series):
            date_cols.append(col)
            parsed_dates = pd.to_datetime(col_series, errors="coerce").dropna()
            if len(parsed_dates) > 0:
                base_profile.update({
                    "min_date": str(parsed_dates.min()),
                    "max_date": str(parsed_dates.max()),
                    "unique_dates": int(parsed_dates.nunique())
                })
            else:
                base_profile.update({
                    "min_date": None, "max_date": None, "unique_dates": 0
                })

        # Categorical / string statistics
        else:
            categorical_cols.append(col)
            clean_str = col_series.dropna().astype(str).str.strip()
            top_counts = clean_str.value_counts().head(5).to_dict()
            most_frequent = clean_str.mode().iloc[0] if not clean_str.empty else None
            
            base_profile.update({
                "top_frequent_values": {str(k): int(v) for k, v in top_counts.items()},
                "mode": str(most_frequent) if most_frequent is not None else None,
                "avg_string_length": round(float(clean_str.str.len().mean()), 1) if not clean_str.empty else 0
            })

        column_profiles[col] = base_profile

    return {
        "row_count": row_count,
        "column_count": col_count,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_columns": date_cols,
        "columns": column_profiles
    }
