"""
cleaner/validators.py
Universal Validators (F2) & Statistical Outlier Detectors:
Produces reviewable FLAG proposals only. Never modifies data.

Universal Validators:
- email: standard RFC-like email pattern
- url: http/https web addresses
- phone: phone number shapes (E.164, US domestic, international)
- postal: US 5-digit/ZIP+4, Canadian postal, UK postal
- boolean: common truth/false representations
- iso_code: ISO 3166-1 alpha-2, alpha-3 country codes, US state postal abbreviations

Outliers:
- IQR method: 1.5x IQR (mild) and 3.0x IQR (extreme) outlier counts per numeric column.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from cleaner.rules.base import Proposal


# Standard Regexes for Universal Validators
EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
URL_RE = re.compile(r"^https?://[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+(/[^\s]*)?$")
PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,25}$")
US_ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")
CA_POSTAL_RE = re.compile(r"^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$")
UK_POSTAL_RE = re.compile(r"^[A-Za-z]{1,2}\d[A-Za-z\d]?[ -]?\d[A-Za-z]{2}$")

BOOLEAN_WORDS = {
    "true", "false", "yes", "no", "y", "n", "t", "f", "1", "0"
}

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP"
}

ISO_COUNTRY_CODES = {
    # Alpha-2
    "US", "CA", "GB", "UK", "DE", "FR", "IT", "ES", "AU", "JP", "CN", "IN",
    "BR", "MX", "RU", "ZA", "KR", "NL", "SE", "NO", "DK", "FI", "CH", "AT",
    "BE", "IE", "NZ", "SG", "HK", "AE", "SA", "PL", "PT", "GR", "TR", "IL",
    # Alpha-3
    "USA", "CAN", "GBR", "DEU", "FRA", "ITA", "ESP", "AUS", "JPN", "CHN", "IND",
    "BRA", "MEX", "RUS", "ZAF", "KOR", "NLD", "SWE", "NOR", "DNK", "FIN", "CHE",
    "AUT", "BEL", "IRL", "NZL", "SGP", "HKG", "ARE", "SAU", "POL", "PRT", "GRC",
    "TUR", "ISR"
}


def is_valid_phone(val: str) -> bool:
    if not PHONE_RE.match(val):
        return False
    digits = sum(1 for c in val if c.isdigit())
    return 7 <= digits <= 15


def is_valid_postal(val: str) -> bool:
    return bool(US_ZIP_RE.match(val) or CA_POSTAL_RE.match(val) or UK_POSTAL_RE.match(val))


def check_universal_validators(
    df: pd.DataFrame
) -> List[Proposal]:
    """
    Check universal format validators across DataFrame columns.
    If a column is mostly-valid (>= 80% matches validator, n >= 10),
    flags invalid cells with FLAG tier (never modifies data).
    """
    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows < 10:
        return proposals

    for col in df.columns:
        series = df[col]
        non_empty = [(idx, str(val).strip()) for idx, val in series.items() if str(val).strip() != ""]
        n_valid = len(non_empty)
        if n_valid < 10:
            continue

        validators = [
            ("email", lambda v: bool(EMAIL_RE.match(v))),
            ("url", lambda v: bool(URL_RE.match(v))),
            ("phone", is_valid_phone),
            ("postal_code", is_valid_postal),
            ("boolean", lambda v: v.lower() in BOOLEAN_WORDS),
            ("iso_code", lambda v: v.upper() in US_STATE_CODES or v.upper() in ISO_COUNTRY_CODES)
        ]

        for val_name, validator_func in validators:
            valid_items = [v for _, v in non_empty if validator_func(v)]
            match_ratio = len(valid_items) / n_valid

            # If column is predominantly of this type (>= 80%) but has invalid entries
            if match_ratio >= 0.80 and match_ratio < 1.0:
                invalid_items = [(idx, v) for idx, v in non_empty if not validator_func(v)]
                if invalid_items:
                    sample_invalids = [v for _, v in invalid_items[:5]]
                    proposals.append(Proposal(
                        id=f"FLAG_validator_{val_name}_{col}",
                        kind="validator_flag",
                        tier="FLAG",
                        columns=[col],
                        description=f"Flag {len(invalid_items)} invalid '{val_name}' formats in column '{col}'",
                        evidence=(
                            f"Column '{col}' exhibits {match_ratio:.1%} adherence to {val_name} format, "
                            f"but {len(invalid_items)} values fail validation. Sample: {sample_invalids}. "
                            f"Flagged for human verification; no cells were changed."
                        ),
                        changes=[]
                    ))
                # Only match the highest confidence validator per column
                break

    return proposals


def check_iqr_outliers(
    df: pd.DataFrame
) -> List[Proposal]:
    """
    Detect statistical outliers in numeric columns using the IQR method.
    Produces column-level FLAG proposals reporting mild (1.5x) and extreme (3.0x) outlier counts.
    Never alters cell values.
    """
    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows < 15:
        return proposals

    for col in df.columns:
        series = df[col]
        # Collect numeric values
        num_vals: List[Tuple[int, float]] = []
        for idx, val in series.items():
            s = str(val).strip()
            if s != "":
                try:
                    num_vals.append((int(idx), float(s)))
                except (ValueError, TypeError):
                    pass

        # Require >= 80% numeric values and >= 15 numbers
        non_empty_count = sum(1 for val in series if str(val).strip() != "")
        if non_empty_count < 15 or len(num_vals) / non_empty_count < 0.80:
            continue

        vals_series = pd.Series([v for _, v in num_vals])
        q1 = vals_series.quantile(0.25)
        q3 = vals_series.quantile(0.75)
        iqr = q3 - q1

        if iqr <= 0:
            continue

        mild_lower = q1 - 1.5 * iqr
        mild_upper = q3 + 1.5 * iqr
        extreme_lower = q1 - 3.0 * iqr
        extreme_upper = q3 + 3.0 * iqr

        mild_outliers = [v for _, v in num_vals if v < mild_lower or v > mild_upper]
        extreme_outliers = [v for _, v in num_vals if v < extreme_lower or v > extreme_upper]

        if mild_outliers:
            proposals.append(Proposal(
                id=f"FLAG_outliers_iqr_{col}",
                kind="outlier_iqr",
                tier="FLAG",
                columns=[col],
                description=f"Flag {len(mild_outliers)} numeric outliers in column '{col}' ({len(extreme_outliers)} extreme)",
                evidence=(
                    f"IQR Analysis: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}. "
                    f"Bounds: 1.5x IQR [{mild_lower:.2f}, {mild_upper:.2f}], "
                    f"3.0x IQR [{extreme_lower:.2f}, {extreme_upper:.2f}]. "
                    f"Found {len(mild_outliers)} mild and {len(extreme_outliers)} extreme outliers. "
                    f"Flagged for data scientist review; values preserved unchanged."
                ),
                changes=[]
            ))

    return proposals
