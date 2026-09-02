import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np

EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
DATE_PATTERNS = [
    r"^\d{4}-\d{1,2}-\d{1,2}$",
    r"^\d{1,2}/\d{1,2}/\d{4}$",
    r"^\d{1,2}-\d{1,2}-\d{4}$",
    r"^\d{4}/\d{1,2}/\d{1,2}$"
]

def detect_column_schema(series: pd.Series) -> Dict[str, Any]:
    """
    Infers logical data type, semantic hints, and structural properties of a column.
    """
    col_name = str(series.name)
    clean_series = series.dropna()
    total_valid = len(clean_series)
    
    if total_valid == 0:
        return {
            "logical_type": "unknown",
            "semantic_hint": "empty",
            "is_nullable": True,
            "sample_values": []
        }
        
    sample = clean_series.head(5).tolist()

    # 1. Native numeric
    if pd.api.types.is_numeric_dtype(series):
        is_id = False
        if ("id" in col_name.lower() or "key" in col_name.lower()) and series.nunique() / total_valid > 0.9:
            is_id = True
        return {
            "logical_type": "numeric",
            "semantic_hint": "identifier" if is_id else ("integer" if pd.api.types.is_integer_dtype(series) else "float"),
            "is_nullable": series.isna().any(),
            "sample_values": sample
        }
        
    # 2. Native datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return {
            "logical_type": "datetime",
            "semantic_hint": "timestamp",
            "is_nullable": series.isna().any(),
            "sample_values": [str(x) for x in sample]
        }

    # 3. Native boolean
    if pd.api.types.is_bool_dtype(series):
        return {
            "logical_type": "boolean",
            "semantic_hint": "flag",
            "is_nullable": series.isna().any(),
            "sample_values": sample
        }

    # String-based inference
    str_series = clean_series.astype(str).str.strip()
    
    # Check for boolean representations
    bool_values = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
    matching_bool = str_series.str.lower().isin(bool_values).sum()
    if matching_bool / total_valid >= 0.9 and series.nunique() <= 4:
        return {
            "logical_type": "boolean",
            "semantic_hint": "flag",
            "is_nullable": series.isna().any(),
            "sample_values": sample
        }

    # Check for potential email
    email_matches = str_series.apply(lambda x: bool(EMAIL_REGEX.match(x))).sum()
    if email_matches / total_valid >= 0.7:
        return {
            "logical_type": "email",
            "semantic_hint": "contact_info",
            "is_nullable": series.isna().any(),
            "sample_values": sample
        }

    # Check for date strings
    date_pattern_matches = 0
    for pat in DATE_PATTERNS:
        matches = str_series.apply(lambda x: bool(re.match(pat, x))).sum()
        if matches / total_valid >= 0.5:
            date_pattern_matches = matches
            break
            
    # Also check if name hints at date
    is_date_named = any(k in col_name.lower() for k in ["date", "time", "dob", "birth", "created", "joined", "timestamp"])
    if date_pattern_matches > 0 or is_date_named:
        # Check parseability
        try:
            parsed = pd.to_datetime(str_series, format="mixed", errors="coerce")
            if parsed.notna().sum() / total_valid >= 0.5:
                return {
                    "logical_type": "datetime",
                    "semantic_hint": "date_string",
                    "is_nullable": series.isna().any(),
                    "sample_values": sample
                }
        except Exception:
            pass

    # Check for string-encoded numeric
    try:
        num_converted = pd.to_numeric(str_series, errors="coerce")
        if num_converted.notna().sum() / total_valid >= 0.8:
            return {
                "logical_type": "numeric",
                "semantic_hint": "text_encoded_numeric",
                "is_nullable": series.isna().any(),
                "sample_values": sample
            }
    except Exception:
        pass

    # Identifier check
    if ("id" in col_name.lower() or "uuid" in col_name.lower() or "code" in col_name.lower()) and series.nunique() / total_valid > 0.9:
        return {
            "logical_type": "identifier",
            "semantic_hint": "key",
            "is_nullable": series.isna().any(),
            "sample_values": sample
        }

    # Categorical vs Free text
    avg_length = str_series.str.len().mean()
    unique_ratio = series.nunique() / total_valid
    
    if unique_ratio < 0.25 or series.nunique() <= 50:
        return {
            "logical_type": "categorical",
            "semantic_hint": "discrete_category",
            "is_nullable": series.isna().any(),
            "sample_values": sample
        }
    elif avg_length > 50:
        return {
            "logical_type": "text",
            "semantic_hint": "long_text",
            "is_nullable": series.isna().any(),
            "sample_values": sample
        }
    else:
        return {
            "logical_type": "categorical",
            "semantic_hint": "high_cardinality_category",
            "is_nullable": series.isna().any(),
            "sample_values": sample
        }


def detect_dataset_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Detects logical types and hints across all columns of a dataframe."""
    return {col: detect_column_schema(df[col]) for col in df.columns}
