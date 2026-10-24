"""
tests/test_universal_enhancements.py
Unit tests verifying:
1. Smart delimiter & header sniffing across formats (',', ';', '\\t', '|')
2. Contextual validator disambiguation guards (ISO vs Boolean, Boolean vs Numeric, Phone vs Float, Postal vs SKU)
3. R2 Dominant unit suffix preservation (Movies_1 duration invariant)
4. Vectorized R6 dependency repair correctness
"""

import pandas as pd
import pytest
from cleaner.io import load_table, sniff_delimiter_and_header
from cleaner.validators import check_universal_validators, check_iqr_outliers, is_valid_phone
from cleaner.rules.r2_numeric import propose_r2_numeric
from cleaner.rules.r6_dependency_repair import propose_r6_dependency_repair


def test_sniff_delimiter_and_header():
    # Semicolon with header
    sample_semi = '"age";"job";"marital"\n30;"unemployed";"married"\n33;"services";"married"'
    sep, hdr = sniff_delimiter_and_header(sample_semi)
    assert sep == ";"
    assert hdr == 0

    # Comma without header (e.g. Adult / WDBC numeric data)
    sample_no_hdr = "39, State-gov, 77516, Bachelors, 13\n50, Self-emp, 83311, Bachelors, 13\n38, Private, 215646, HS-grad, 9"
    sep, hdr = sniff_delimiter_and_header(sample_no_hdr)
    assert sep == ","
    assert hdr is None

    # Standard comma with header
    sample_std = "id,name,salary\n1,Alice,50000\n2,Bob,60000"
    sep, hdr = sniff_delimiter_and_header(sample_std)
    assert sep == ","
    assert hdr == 0


def test_validator_boolean_vs_iso_code():
    """Ensure column with 'yes' and 'no' is not flagged as Norway ISO Country Code."""
    df = pd.DataFrame({
        "default": ["no"] * 90 + ["yes"] * 10
    })
    props = check_universal_validators(df)
    prop_ids = [p.id for p in props]
    assert not any("iso_code" in pid for pid in prop_ids)


def test_validator_zero_inflated_not_boolean():
    """Ensure zero-inflated numeric column like capital-gain (0s and positive earnings) is not flagged as boolean."""
    df = pd.DataFrame({
        "capital_gain": ["0"] * 90 + ["2174", "14084", "5178", "1000", "5000", "2000", "3000", "4000", "1500", "2500"]
    })
    props = check_universal_validators(df)
    prop_ids = [p.id for p in props]
    assert not any("boolean" in pid for pid in prop_ids)


def test_validator_phone_vs_float():
    """Ensure floating-point scientific measurements are not classified as phone numbers."""
    assert not is_valid_phone("25.38")
    assert not is_valid_phone("0.006399")
    assert not is_valid_phone("1001.0")
    assert is_valid_phone("+1 555-123-4567")
    assert is_valid_phone("555-0199")


def test_validator_postal_vs_alphanumeric_skus():
    """Ensure alphanumeric product SKUs (e.g. 85123A) do not trigger US postal code validator."""
    df = pd.DataFrame({
        "StockCode": ["85123A", "71053", "84406B", "22423", "85099B"] * 20
    })
    props = check_universal_validators(df)
    prop_ids = [p.id for p in props]
    assert not any("postal_code" in pid for pid in prop_ids)


def test_r2_dominant_unit_preservation():
    """Ensure column where >= 85% of values have 'min' preserves the unit instead of stripping it."""
    # 95 rows with 'min', 5 rows with 'hr' or no unit
    vals = [f"{i} min" for i in range(50, 145)] + ["100 min.", "90", "120 min", "110 min", "105 min"]
    df = pd.DataFrame({"duration": vals})
    props = propose_r2_numeric(df)
    canonical_props = [p for p in props if p.kind == "R2_numeric"]
    # Check that it doesn't strip 'min' from the 95 already correct values
    if canonical_props:
        for p in canonical_props:
            for c in p.changes:
                # Valid target should include 'min'
                assert "min" in c.new_value


def test_r6_vectorized_dependency_repair():
    """Ensure vectorized R6 dependency repair correctly fixes violations and fills blanks."""
    rows = []
    for i in range(25):
        if i == 5:
            city = "Ny City"  # typo
        elif i == 6:
            city = ""  # blank
        else:
            city = "New York"
        rows.append({"zip": "10001", "city": city})

    for i in range(15):
        rows.append({"zip": "90210", "city": "Beverly Hills"})

    df = pd.DataFrame(rows)
    props = propose_r6_dependency_repair(df)
    assert len(props) > 0
    prop = props[0]
    assert prop.kind == "R6_dependency_repair"
    has_repair = any(c.old_value == "Ny City" and c.new_value == "New York" for c in prop.changes)
    has_fill = any(c.old_value == "" and c.new_value == "New York" for c in prop.changes)
    assert has_repair
    assert has_fill
