"""
cleaner/io.py
Strict string-only ingestion and export with encoding fallback.
Invariants:
- All values loaded strictly as str.
- keep_default_na=False: No auto-conversion of NA, null, nan, None to np.nan.
- No automatic type inference: Leading zeros in IDs, ZIP codes, and text codes remain intact.
- Untouched cells stay byte-identical.
"""

import os
import io
from typing import Union, List, Optional
import pandas as pd


ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]


def load_table(
    source: Union[str, os.PathLike, io.BytesIO, io.StringIO],
    encoding: Optional[str] = None
) -> pd.DataFrame:
    """
    Load a CSV or Excel table strictly as string data without type inference.
    
    Parameters:
      source: File path, path-like object, or file-like buffer.
      encoding: Optional specific encoding. If None, tries standard fallback chain.
      
    Returns:
      pd.DataFrame where every column has dtype=object and all values are Python str.
    """
    is_path = isinstance(source, (str, os.PathLike))
    file_ext = ""
    if is_path:
        file_ext = os.path.splitext(str(source))[1].lower()

    # Excel loading
    if file_ext in [".xlsx", ".xls"] or hasattr(source, "name") and str(getattr(source, "name", "")).endswith((".xlsx", ".xls")):
        df = pd.read_excel(source, dtype=str, keep_default_na=False, engine="openpyxl")
        return df.fillna("").astype(str)

    # Buffer or CSV loading with encoding fallback
    if encoding is not None:
        encodings = [encoding] + [e for e in ENCODINGS_TO_TRY if e != encoding]
    else:
        encodings = ENCODINGS_TO_TRY

    last_error = None

    if is_path:
        for enc in encodings:
            try:
                df = pd.read_csv(
                    source,
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                    encoding=enc,
                    low_memory=False
                )
                return df.fillna("").astype(str)
            except (UnicodeDecodeError, LookupError) as e:
                last_error = e
                continue
    else:
        # Buffer
        content = source.read() if hasattr(source, "read") else source
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        for enc in encodings:
            try:
                text_stream = io.StringIO(content_bytes.decode(enc))
                df = pd.read_csv(
                    text_stream,
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                    low_memory=False
                )
                return df.fillna("").astype(str)
            except (UnicodeDecodeError, LookupError) as e:
                last_error = e
                continue

    if last_error:
        raise ValueError(f"Failed to decode file with encodings {encodings}: {last_error}")
    raise ValueError(f"Unable to load table from source: {source}")


def save_table(df: pd.DataFrame, path: Union[str, os.PathLike]) -> None:
    """
    Save DataFrame strictly preserving strings without index.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False, engine="openpyxl")
    else:
        df.to_csv(path, index=False, encoding="utf-8")
