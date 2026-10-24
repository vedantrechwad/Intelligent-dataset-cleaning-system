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


FLOAT_RE = re.compile(r"^[-+]?\d+\.\d+$")


def is_valid_phone(val: str) -> bool:
    if FLOAT_RE.match(val):
        return False
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
    Incorporates contextual disambiguation guards to avoid false positive traps.
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

        # Contextual guards:
        sample_vals = [v for _, v in non_empty[:200]]
        has_non_binary_numbers = any(v.isdigit() and int(v) > 1 for v in sample_vals)
        has_boolean_context = any(v.lower() in {"yes", "true", "false", "y", "no"} for v in sample_vals)
        has_alphanumeric_skus = sum(1 for v in sample_vals if bool(re.search(r"[a-zA-Z]", v) and re.search(r"\d", v))) / len(sample_vals) > 0.05

        effective_boolean_words = (BOOLEAN_WORDS - {"0", "1"}) if has_non_binary_numbers else BOOLEAN_WORDS

        def is_iso_code(v: str) -> bool:
            up = v.upper()
            if has_boolean_context and up == "NO":
                return False
            return up in US_STATE_CODES or up in ISO_COUNTRY_CODES

        validators = [
            ("email", lambda v: bool(EMAIL_RE.match(v))),
            ("url", lambda v: bool(URL_RE.match(v))),
            ("phone", is_valid_phone),
            ("postal_code", lambda v: False if has_alphanumeric_skus else is_valid_postal(v)),
            ("boolean", lambda v: v.lower() in effective_boolean_words),
            ("iso_code", is_iso_code)
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
    Detect statistical outliers in numeric columns using fast vectorized IQR method.
    Produces column-level FLAG proposals reporting mild (1.5x) and extreme (3.0x) outlier counts.
    Never alters cell values.
    """
    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows < 15:
        return proposals

    for col in df.columns:
        series = df[col]
        s_stripped = series.astype(str).str.strip()
        non_empty_mask = s_stripped != ""
        non_empty_count = int(non_empty_mask.sum())
        if non_empty_count < 15:
            continue

        num_s = pd.to_numeric(s_stripped[non_empty_mask], errors="coerce")
        valid_num = num_s.dropna()
        if len(valid_num) < 15 or (len(valid_num) / non_empty_count) < 0.80:
            continue

        q1 = float(valid_num.quantile(0.25))
        q3 = float(valid_num.quantile(0.75))
        iqr = q3 - q1

        if iqr <= 0:
            continue

        mild_lower = q1 - 1.5 * iqr
        mild_upper = q3 + 1.5 * iqr
        extreme_lower = q1 - 3.0 * iqr
        extreme_upper = q3 + 3.0 * iqr

        mild_mask = (valid_num < mild_lower) | (valid_num > mild_upper)
        extreme_mask = (valid_num < extreme_lower) | (valid_num > extreme_upper)

        mild_count = int(mild_mask.sum())
        extreme_count = int(extreme_mask.sum())

        if mild_count > 0:
            proposals.append(Proposal(
                id=f"FLAG_outliers_iqr_{col}",
                kind="outlier_iqr",
                tier="FLAG",
                columns=[col],
                description=f"Flag {mild_count} numeric outliers in column '{col}' ({extreme_count} extreme)",
                evidence=(
                    f"IQR Analysis: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}. "
                    f"Mild outlier bounds: [{mild_lower:.2f}, {mild_upper:.2f}]. "
                    f"Extreme bounds: [{extreme_lower:.2f}, {extreme_upper:.2f}]. "
                    f"Found {mild_count} mild and {extreme_count} extreme outliers. Flagged for review; values untouched."
                ),
                changes=[]
            ))

    return proposals
