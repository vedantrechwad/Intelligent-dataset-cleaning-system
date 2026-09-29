"""
tests/test_phase1.py
Comprehensive unit and property tests for Phase 1:
- io.py string loading, leading-zero preservation, no auto-na
- R1 missing tokens (rarity threshold, frequent to review, empty-after-trim)
- R2 numeric canonicalization (units, %, descriptors, no rescaling)
- R3 whitespace & case (trimming, collapse, dominant casing, short codes protection)
- Engine invariants: idempotency, untouched cell identity, no change outside approved rules
- Edge cases: empty col, all-missing col, 1-row file, unicode, long strings
"""

import io
import pytest
import pandas as pd
from cleaner.io import load_table, save_table
from cleaner.rules.base import Proposal, CellChange
from cleaner.rules.r1_missing import propose_r1_missing
from cleaner.rules.r2_numeric import propose_r2_numeric
from cleaner.rules.r3_whitespace_case import propose_r3_whitespace_case
from cleaner.engine import CleaningEngine


def test_io_pure_string_and_leading_zeros():
    csv_data = "id,code,val\n00145,NA,12.0\n00146,null,15.5\n00147,,\n"
    df = load_table(io.StringIO(csv_data))

    assert df.shape == (3, 3)
    # Leading zeros preserved
    assert df.at[0, "id"] == "00145"
    assert df.at[1, "id"] == "00146"
    assert df.at[2, "id"] == "00147"
    # Disguised tokens kept as strings, not np.nan
    assert df.at[0, "code"] == "NA"
    assert df.at[1, "code"] == "null"
    assert df.at[2, "code"] == ""
    assert isinstance(df.at[0, "val"], str)


def test_r1_missing_rare_vs_frequent():
    # 100 rows: col 'rare_na' has 2 'NA' (2% <= 5%), col 'freq_na' has 10 'NA' (10% > 5%)
    rows = []
    for i in range(100):
        r_na = "NA" if i < 2 else "valid"
        f_na = "NA" if i < 10 else "valid"
        rows.append({"rare_na": r_na, "freq_na": f_na, "trim_me": "   " if i == 0 else "text"})
    df = pd.DataFrame(rows)

    proposals = propose_r1_missing(df)
    prop_map = {p.id: p for p in proposals}

    # Rare should be AUTO
    assert "R1_missing_rare_na_NA" in prop_map
    assert prop_map["R1_missing_rare_na_NA"].tier == "AUTO"
    assert prop_map["R1_missing_rare_na_NA"].n_cells == 2

    # Frequent should be REVIEW
    assert "R1_missing_freq_na_NA" in prop_map
    assert prop_map["R1_missing_freq_na_NA"].tier == "REVIEW"
    assert prop_map["R1_missing_freq_na_NA"].n_cells == 10

    # Whitespace-only should be AUTO
    assert "R1_missing_trim_trim_me" in prop_map
    assert prop_map["R1_missing_trim_trim_me"].tier == "AUTO"
    assert prop_map["R1_missing_trim_trim_me"].n_cells == 1


def test_r1_edge_cases():
    # 1. Empty dataframe
    empty_df = pd.DataFrame(columns=["a", "b"])
    assert propose_r1_missing(empty_df) == []

    # 2. All-missing column
    all_na = pd.DataFrame({"col": ["NA", "NA", "NA"]})
    p = propose_r1_missing(all_na)
    assert len(p) == 1
    assert p[0].tier == "REVIEW"  # 100% frequency -> REVIEW

    # 3. 1-row file
    one_row = pd.DataFrame({"code": ["007"], "name": ["Bond"]})
    assert propose_r1_missing(one_row) == []


def test_r2_numeric_canonicalization():
    # Column with ounces: 8 rows with unit, 2 with descriptor text
    data = {
        "ounces": [
            "12.0 oz.", "12.0 oz", "12.0 ounce", "16.0 oz.", "16.0 oz",
            "12.0 OZ.", "24.0 oz.", "8.4 oz.", "12.0 oz. Alumi-Tek", "16.0 oz. Silo Can"
        ]
    }
    df = pd.DataFrame(data)
    proposals = propose_r2_numeric(df)
    tiers = {p.id: p.tier for p in proposals}

    # Standard unit tokens stripped -> AUTO
    assert "R2_numeric_canonical_ounces" in tiers
    assert tiers["R2_numeric_canonical_ounces"] == "AUTO"

    # Descriptor text dropped -> REVIEW
    assert "R2_numeric_descriptor_ounces" in tiers
    assert tiers["R2_numeric_descriptor_ounces"] == "REVIEW"


def test_r2_percentage_ambiguous_scale():
    # Column with mixed % and plain numbers
    data = {
        "rate": ["0.05", "0.06", "0.04", "0.07", "0.09%", "0.08%"]
    }
    df = pd.DataFrame(data)
    proposals = propose_r2_numeric(df)

    assert len(proposals) == 1
    p = proposals[0]
    assert p.tier == "REVIEW"
    assert "ambiguous" in p.description.lower()
    # Ensure numeric value preserved, not rescaled
    assert p.changes[0].new_value == "0.09"


def test_r2_non_numeric_address_preserved():
    data = {
        "address": ["123 Main St", "456 Elm Ave", "789 Oak Rd", "101 Pine Blvd", "202 Maple Dr"]
    }
    df = pd.DataFrame(data)
    proposals = propose_r2_numeric(df)
    assert proposals == []


def test_r3_whitespace_and_case():
    data = {
        "gender": ["Female", "Female", "Female", "female", " Male ", "Male"],
        "state": ["MI", "MI", "MI", "IN", "IN", "OR"],  # Short codes must NOT merge
    }
    df = pd.DataFrame(data)
    proposals = propose_r3_whitespace_case(df)
    prop_map = {p.id: p for p in proposals}

    # Gender should have whitespace and casing proposals
    assert "R3_case_gender" in prop_map
    assert prop_map["R3_case_gender"].changes[0].old_value == "female"
    assert prop_map["R3_case_gender"].changes[0].new_value == "Female"

    # State short codes MI vs IN must NOT have any proposal
    assert "R3_case_state" not in prop_map


def test_engine_invariants_and_idempotency():
    data = {
        "id": ["001", "002", "003"],
        "name": [" Alice ", "Bob", "bob"],
        "units": ["12.0 oz.", "16.0 oz.", "12.0"],
        "flag": ["NA", "valid", "valid"]  # 1/3 = 33% -> REVIEW
    }
    df = pd.DataFrame(data)
    engine = CleaningEngine()

    proposals = engine.generate_proposals(df)
    auto_proposals = [p for p in proposals if p.tier == "AUTO"]

    cleaned_df, diffs = engine.apply(df, auto_proposals)

    # Invariant 1: Untouched cells stay identical
    assert cleaned_df.at[0, "id"] == "001"
    assert cleaned_df.at[0, "flag"] == "NA"  # Review tier was not approved

    # Invariant 2: Changes only where approved
    for d in diffs:
        assert d["tier"] == "AUTO"

    # Invariant 3: Idempotency - applying engine again produces zero diffs
    proposals_pass2 = engine.generate_proposals(cleaned_df)
    auto_pass2 = [p for p in proposals_pass2 if p.tier == "AUTO"]
    cleaned_df_pass2, diffs_pass2 = engine.apply(cleaned_df, auto_pass2)

    assert diffs_pass2 == []
    pd.testing.assert_frame_equal(cleaned_df, cleaned_df_pass2)


def test_long_string_and_unicode():
    data = {
        "text": [" café ", "café", "  very " + "long_" * 100 + "string  "]
    }
    df = pd.DataFrame(data)
    engine = CleaningEngine()
    props = engine.generate_proposals(df)
    cleaned, diffs = engine.apply(df, [p for p in props if p.tier == "AUTO"])

    assert cleaned.at[0, "text"] == "café"
    assert not cleaned.at[2, "text"].startswith(" ")
    assert not cleaned.at[2, "text"].endswith(" ")
