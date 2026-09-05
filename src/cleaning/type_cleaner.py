import re
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

BOOLEAN_MAPPING = {
    "yes": True, "no": False,
    "true": True, "false": False,
    "t": True, "f": False,
    "1": True, "0": False,
    "y": True, "n": False
}

NULL_TOKENS = {"n/a", "na", "null", "none", "?", "-", "undefined", "missing", ""}

def clean_data_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Safely converts columns with predominantly numeric content into proper numeric dtypes,
    converts stringified numbers, normalizes booleans, and nullifies unknown tokens.
    """
    cleaned_df = df.copy(deep=True)
    logs = []

    for col in cleaned_df.columns:
        series = cleaned_df[col]
        valid_series = series.dropna()
        if len(valid_series) == 0:
            continue

        if not pd.api.types.is_numeric_dtype(series):
            # Evaluate properties
            str_series = valid_series.astype(str).str.strip().str.lower()
            is_predominantly_bool = str_series.isin(BOOLEAN_MAPPING.keys()).mean() > 0.8
            
            num_converted = pd.to_numeric(valid_series, errors="coerce")
            is_predominantly_numeric = (num_converted.notna().sum() / len(valid_series)) >= 0.70

            for row_idx, val in series.items():
                if pd.isna(val) or val is None:
                    continue
                
                str_val = str(val).strip()
                lower_val = str_val.lower()

                # Null-tokens
                if lower_val in NULL_TOKENS:
                    cleaned_df.at[row_idx, col] = np.nan
                    logs.append({
                        "row": int(row_idx),
                        "column": col,
                        "original_value": str_val,
                        "issue_type": "type_inconsistency",
                        "action": "converted_to_null",
                        "corrected_value": None,
                        "confidence": 0.99,
                        "reason": f"Value '{str_val}' safely converted to NaN"
                    })
                    continue

                # Boolean normalization
                if is_predominantly_bool and not pd.api.types.is_bool_dtype(series):
                    if lower_val in BOOLEAN_MAPPING:
                        clean_bool = BOOLEAN_MAPPING[lower_val]
                        if str_val != str(clean_bool):
                            cleaned_df.at[row_idx, col] = clean_bool
                            logs.append({
                                "row": int(row_idx),
                                "column": col,
                                "original_value": str_val,
                                "issue_type": "type_inconsistency",
                                "action": "converted_to_boolean",
                                "corrected_value": clean_bool,
                                "confidence": 0.99,
                                "reason": f"Safely normalized '{str_val}' to {clean_bool}"
                            })
                        continue

                # Numeric normalization
                if is_predominantly_numeric:
                    clean_str = re.sub(r'[$,€£]', '', str_val)
                    if clean_str.count(',') >= 1 and clean_str.count('.') <= 1:
                        clean_str = clean_str.replace(',', '')
                    try:
                        num = float(clean_str)
                        clean_num = int(num) if num.is_integer() else num
                        if str(val) != str(clean_num):
                            cleaned_df.at[row_idx, col] = clean_num
                            logs.append({
                                "row": int(row_idx),
                                "column": col,
                                "original_value": str(val),
                                "issue_type": "type_inconsistency",
                                "action": "converted_to_numeric",
                                "corrected_value": clean_num,
                                "confidence": 0.99,
                                "reason": f"Safely parsed string '{val}' as numeric {clean_num}"
                            })
                    except (ValueError, TypeError):
                        pass

            # Cast column dtype if perfectly clean
            if is_predominantly_bool:
                try:
                    cleaned_df[col] = cleaned_df[col].astype("boolean")
                except Exception:
                    pass
            elif is_predominantly_numeric:
                try:
                    if cleaned_df[col].notna().all() and (cleaned_df[col].astype(str).str.isnumeric().all()):
                       pass
                    else:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                except Exception:
                    pass

    return cleaned_df, logs
