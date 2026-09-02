from typing import Tuple, List, Dict, Any
import pandas as pd

def clean_data_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Safely converts columns with predominantly numeric content into proper numeric dtypes,
    and converts stringified numbers into real numbers.
    """
    cleaned_df = df.copy(deep=True)
    logs = []

    for col in cleaned_df.columns:
        series = cleaned_df[col]
        valid_series = series.dropna()
        if len(valid_series) == 0:
            continue

        if not pd.api.types.is_numeric_dtype(series):
            # Check if values are safely convertible to numbers
            num_converted = pd.to_numeric(valid_series, errors="coerce")
            valid_num_count = num_converted.notna().sum()
            ratio = valid_num_count / len(valid_series)

            if ratio >= 0.85:
                # Safely convert eligible cells
                for row_idx, val in series.items():
                    if pd.isna(val) or val is None:
                        continue
                    try:
                        num = float(str(val).strip())
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

                # If all non-null values are numeric, cast column
                if num_converted.notna().all():
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                    except Exception:
                        pass

    return cleaned_df, logs
