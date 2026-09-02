import os
import io
import copy
from typing import Tuple, Dict, Any, Union
import pandas as pd
from src.utils.helpers import logger

def load_dataset(file_source: Union[str, io.BytesIO], filename: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Safely loads a CSV or XLSX dataset.
    Preserves the original dataset completely without modification.
    Creates a separate working copy.
    
    Returns:
        (original_df, working_df, metadata)
    """
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise FileNotFoundError(f"File not found: {file_source}")
        ext = os.path.splitext(file_source)[1].lower()
        fname = os.path.basename(file_source)
    else:
        fname = filename or "uploaded_dataset"
        ext = os.path.splitext(fname)[1].lower()
        
    try:
        if ext in [".csv", ".txt"]:
            # Try reading with utf-8 first, fallback to latin-1
            try:
                if isinstance(file_source, str):
                    df = pd.read_csv(file_source, encoding="utf-8")
                else:
                    file_source.seek(0)
                    df = pd.read_csv(file_source, encoding="utf-8")
            except (UnicodeDecodeError, Exception):
                if isinstance(file_source, str):
                    df = pd.read_csv(file_source, encoding="latin-1")
                else:
                    file_source.seek(0)
                    df = pd.read_csv(file_source, encoding="latin-1")
                    
        elif ext in [".xlsx", ".xls"]:
            if isinstance(file_source, str):
                df = pd.read_excel(file_source, engine="openpyxl")
            else:
                file_source.seek(0)
                df = pd.read_excel(file_source, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file format: '{ext}'. Supported formats are CSV and XLSX.")
            
    except Exception as e:
        logger.error(f"Error reading dataset {fname}: {e}")
        raise ValueError(f"Could not parse file '{fname}': {str(e)}")

    if df.empty:
        raise ValueError(f"Dataset '{fname}' is empty (0 rows or 0 columns).")

    # Strip column names of leading/trailing whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Create immutable original copy and mutable working copy
    original_df = df.copy(deep=True)
    working_df = df.copy(deep=True)

    metadata = {
        "filename": fname,
        "format": ext.replace(".", "").upper(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum())
    }
    
    logger.info(f"Loaded {fname}: {metadata['row_count']} rows, {metadata['column_count']} cols.")
    return original_df, working_df, metadata
