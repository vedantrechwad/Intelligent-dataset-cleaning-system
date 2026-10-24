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


import re
from typing import Union, List, Optional, Tuple


ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]


def sniff_delimiter_and_header(sample_text: str) -> Tuple[str, Optional[int]]:
    """
    Automatically detect CSV/table delimiter (',', ';', '\\t', '|')
    and whether row 0 represents column names or raw data.
    """
    lines = [l.strip() for l in sample_text.splitlines() if l.strip()][:10]
    if not lines:
        return ",", 0

    counts = {",": 0, ";": 0, "\t": 0, "|": 0}
    for l in lines:
        for delim in counts:
            counts[delim] += l.count(delim)

    best_delim = max(counts, key=counts.get)
    sep = best_delim if counts[best_delim] > 0 else ","

    if len(lines) < 2:
        return sep, 0

    first_tokens = [t.strip().strip("'\"") for t in lines[0].split(sep)]
    num_like_first = sum(1 for t in first_tokens if re.match(r"^[-+]?\d+(\.\d+)?$", t))
    num_ratio_first = num_like_first / max(1, len(first_tokens))

    # If first row has >= 20% numbers, row 0 is almost certainly data (not header)
    if num_ratio_first >= 0.20:
        return sep, None

    return sep, 0


def load_table(
    source: Union[str, os.PathLike, io.BytesIO, io.StringIO],
    encoding: Optional[str] = None,
    sep: Optional[str] = None,
    header: Optional[Union[int, str]] = "infer",
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load a CSV, text, or Excel table strictly as string data without type inference.
    Supports automatic delimiter sniffing (',', ';', '\\t', '|'),
    header detection, and encoding fallbacks.
    
    Parameters:
      source: File path, path-like object, or file-like buffer.
      encoding: Optional specific encoding. If None, tries standard fallback chain.
      sep: Optional delimiter override. If None, auto-sniffed.
      header: Row to use as header, None for headerless, or 'infer' for auto-detection.
      
    Returns:
      pd.DataFrame where every column has dtype=object and all values are Python str.
    """
    is_path = isinstance(source, (str, os.PathLike))
    file_ext = ""
    if is_path:
        file_ext = os.path.splitext(str(source))[1].lower()

    # Excel loading
    if file_ext in [".xlsx", ".xls"] or hasattr(source, "name") and str(getattr(source, "name", "")).endswith((".xlsx", ".xls")):
        engine = "xlrd" if (file_ext == ".xls" or str(getattr(source, "name", "")).endswith(".xls")) else "openpyxl"
        df = pd.read_excel(source, dtype=str, keep_default_na=False, engine=engine, nrows=nrows)
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
                # Sniff delimiter & header from first 16KB if needed
                detected_sep = sep
                detected_header = header
                if detected_sep is None or detected_header == "infer":
                    with open(source, "r", encoding=enc, errors="ignore") as fp:
                        sample_chunk = fp.read(16384)
                    sniffed_sep, sniffed_hdr = sniff_delimiter_and_header(sample_chunk)
                    if detected_sep is None:
                        detected_sep = sniffed_sep
                    if detected_header == "infer":
                        detected_header = sniffed_hdr

                df = pd.read_csv(
                    source,
                    sep=detected_sep,
                    header=detected_header,
                    nrows=nrows,
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                    encoding=enc,
                    low_memory=False
                )
                if detected_header is None:
                    df.columns = [f"col_{i}" for i in range(len(df.columns))]
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
                text_content = content_bytes.decode(enc)
                detected_sep = sep
                detected_header = header
                if detected_sep is None or detected_header == "infer":
                    sample_chunk = text_content[:16384]
                    sniffed_sep, sniffed_hdr = sniff_delimiter_and_header(sample_chunk)
                    if detected_sep is None:
                        detected_sep = sniffed_sep
                    if detected_header == "infer":
                        detected_header = sniffed_hdr

                text_stream = io.StringIO(text_content)
                df = pd.read_csv(
                    text_stream,
                    sep=detected_sep,
                    header=detected_header,
                    nrows=nrows,
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                    low_memory=False
                )
                if detected_header is None:
                    df.columns = [f"col_{i}" for i in range(len(df.columns))]
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
