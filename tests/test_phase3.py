"""
tests/test_phase3.py
Unit and property tests for Phase 3:
- R4 Date canonicalization (unambiguous -> AUTO, ambiguous -> FLAG)
- R7 Category near-miss variants (REVIEW tier, 10x ratio, length >= 6, whole-word guard)
- Universal format validators & IQR outlier detectors (FLAG tier only)
- Upgraded DataCorruptionEngine realistic corruptions & traps
"""

import pytest
import pandas as pd
import numpy as np
from cleaner.rules.r4_dates import propose_r4_dates
from cleaner.rules.r7_category_variants import (
    propose_r7_category_variants,
    is_valid_near_miss,
    levenshtein_distance
)
from cleaner.validators import check_universal_validators, check_iqr_outliers
from cleaner.engine import CleaningEngine
from src.evaluation.corruption_engine import DataCorruptionEngine


def test_r4_dates_unambiguous():
    # Day > 12 in first position -> DD/MM/YYYY
    dates = [
        "25/01/2021",
        "18/02/2021",
        "05/03/2021",
        "22/04/2021",
        "14/05/2021",
        "09/06/2021"
    ]
    df = pd.DataFrame({"order_date": dates})
    props = propose_r4_dates(df)
    assert len(props) == 1
    p = props[0]
    assert p.tier == "AUTO"
    assert p.kind == "R4_dates"
    # Check that 25/01/2021 becomes 2021-01-25
    c0 = [c for c in p.changes if c.row == 0][0]
    assert c0.new_value == "2021-01-25"
    # Check that 05/03/2021 becomes 2021-03-05
    c2 = [c for c in p.changes if c.row == 2][0]
    assert c2.new_value == "2021-03-05"


def test_r4_dates_ambiguous_flagged():
    # All day and month components <= 12 -> Ambiguous (MM/DD vs DD/MM cannot be resolved)
    dates = [
        "01/02/2021",
        "03/04/2021",
        "05/06/2021",
        "07/08/2021",
        "09/10/2021",
        "11/12/2021"
    ]
    df = pd.DataFrame({"transaction_date": dates})
    props = propose_r4_dates(df)
    assert len(props) == 1
    p = props[0]
    assert p.tier == "FLAG"
    assert len(p.changes) == 0  # Crucial safety: ambiguous dates are NEVER changed!


def test_r7_category_variants_guards():
    # 1. Whole-word guard: North Carolina vs South Carolina must NEVER merge
    valid, reason, _ = is_valid_near_miss("North Carolina", "South Carolina")
    assert not valid
    assert "semantic_distinction" in reason or "multiple_words" in reason or "differ" in reason

    # 2. Opposite word guard: East vs West
    valid, reason, _ = is_valid_near_miss("East Carolina", "West Carolina")
    assert not valid

    # 3. Short word guard: length < 6
    valid, reason, _ = is_valid_near_miss("corn", "porn")
    assert not valid
    assert "length_too_short" in reason

    # 4. Valid near-miss typo in long word (edit distance <= 2, length >= 6)
    valid, reason, dist = is_valid_near_miss("Hospitl", "Hospital")
    assert valid
    assert dist == 1

    # 5. Valid minor punctuation/whitespace difference
    valid, reason, _ = is_valid_near_miss("St. Louis", "St Louis")
    assert valid


def test_r7_proposal_generation():
    # Build dataset where 'Hospital' appears 50 times and 'Hospitl' appears 1 time
    rows = ["Hospital"] * 50 + ["Hospitl"] * 1 + ["Clinic"] * 20
    df = pd.DataFrame({"facility_type": rows})
    props = propose_r7_category_variants(df)

    assert len(props) == 1
    p = props[0]
    assert p.tier == "REVIEW"  # Must always be REVIEW
    assert p.kind == "R7_category_variants"
    assert len(p.changes) == 1
    assert p.changes[0].old_value == "Hospitl"
    assert p.changes[0].new_value == "Hospital"


def test_universal_validators():
    # 9 valid emails, 1 invalid email
    emails = [f"user{i}@example.com" for i in range(9)] + ["invalid-email-at-domain"]
    df = pd.DataFrame({"contact_email": emails})
    props = check_universal_validators(df)
    assert len(props) == 1
    p = props[0]
    assert p.tier == "FLAG"
    assert len(p.changes) == 0
    assert "contact_email" in p.columns


def test_iqr_outliers():
    # Mostly normal numbers around 50, with one extreme outlier at 5000
    nums = [50.0 + (i % 5) for i in range(20)] + [5000.0]
    df = pd.DataFrame({"metric": [str(x) for x in nums]})
    props = check_iqr_outliers(df)
    assert len(props) == 1
    p = props[0]
    assert p.tier == "FLAG"
    assert len(p.changes) == 0
    assert "metric" in p.columns
    assert "outlier" in p.description.lower()


def test_upgraded_corruption_engine():
    df = pd.DataFrame({
        "Amount": ["100.0", "150.0", "200.0", "250.0", "300.0"] * 10,
        "City": ["Springfield", "Shelbyville", "Capital City", "Ogdenville", "North Haverbrook"] * 10,
        "JoinDate": ["2022-01-15", "2022-02-18", "2022-03-25", "2022-04-14", "2022-05-20"] * 10
    })
    corrupter = DataCorruptionEngine(seed=42)
    corrupted_df, ground_truth = corrupter.inject_realistic_corruptions(df, cell_rate=0.10, include_traps=True)

    assert len(corrupted_df) == len(df)
    assert len(ground_truth) > 0
    # Trap column should exist
    assert "TRAP_STATE" in corrupted_df.columns
    assert "TRAP_LEGIT_BLANK" in corrupted_df.columns


def test_engine_phase3_integration():
    # Ensure CleaningEngine generates proposals for R4, R7, and Validators without error
    dates = ["25/01/2021", "18/02/2021", "05/03/2021", "22/04/2021", "14/05/2021", "09/06/2021"] * 5
    hospitals = ["Hospital"] * 29 + ["Hospitl"]
    df = pd.DataFrame({
        "record_date": dates,
        "type": hospitals
    })
    engine = CleaningEngine()
    props = engine.generate_proposals(df)
    prop_kinds = {p.kind for p in props}
    assert "R4_dates" in prop_kinds
    assert "R7_category_variants" in prop_kinds

    # In AUTO mode, only R4 should be approved; R7 is REVIEW, so it shouldn't be applied in AUTO
    auto_props = [p for p in props if p.tier == "AUTO"]
    cleaned_auto, _ = engine.apply(df, auto_props)
    # R7 variant was not applied in AUTO mode
    assert "Hospitl" in cleaned_auto["type"].values

    # In AUTO+REVIEW mode, R7 is approved
    review_props = [p for p in props if p.tier in ["AUTO", "REVIEW"]]
    cleaned_review, diffs = engine.apply(df, review_props)
    assert "Hospitl" not in cleaned_review["type"].values
    assert any(d["rule_kind"] == "R7_category_variants" for d in diffs)
