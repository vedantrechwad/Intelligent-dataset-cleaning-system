"""
tests/test_phase2.py
Comprehensive unit and property tests for Phase 2:
- profile.py (shape signatures, minority shapes, missing counts, dependencies)
- R5 compound fields (learning frequent tokens, split, blank fill, safety)
- R6 dependency repair (confidence >= 0.95, majority support >= 2, share >= 2/3, blank fill)
- R8 exact duplicates (detecting duplicates, preserving first occurrence, row drop)
- Engine invariants and idempotency across Phase 2 rules
"""

import pytest
import pandas as pd
from cleaner.profile import profile_table, compute_shape_signature, discover_candidate_dependencies
from cleaner.rules.r5_compound_split import propose_r5_compound_split
from cleaner.rules.r6_dependency_repair import propose_r6_dependency_repair
from cleaner.rules.r8_exact_duplicates import propose_r8_exact_duplicates
from cleaner.engine import CleaningEngine


def test_shape_signatures_and_profiler():
    assert compute_shape_signature("12.0 oz.") == "##.# aa."
    assert compute_shape_signature("4080-OGPJL") == "####-aaaaa"
    assert compute_shape_signature("2024-01-15") == "####-##-##"

    data = {
        "num": ["10", "20", "30", "40", "N/A"],
        "text": ["apple", "banana", "cherry", "date", "fig"]
    }
    df = pd.DataFrame(data)
    profile = profile_table(df)

    assert "num" in profile["columns"]
    assert "text" in profile["columns"]
    assert profile["columns"]["num"]["missing_token_counts"].get("N/A") == 1
    assert profile["columns"]["num"]["numeric_parse_rate"] == 0.8


def test_r5_compound_split_basic():
    # 12 rows: state has 'CA' and 'NY' (each appearing >= 3 times)
    rows = [
        {"city": "San Francisco CA", "state": ""},
        {"city": "San Jose CA", "state": ""},
        {"city": "Oakland CA", "state": ""},
        {"city": "Sacramento CA", "state": "CA"},
        {"city": "Fresno CA", "state": "CA"},
        {"city": "San Diego CA", "state": "CA"},
        {"city": "New York NY", "state": "NY"},
        {"city": "Buffalo NY", "state": "NY"},
        {"city": "Albany NY", "state": "NY"},
        {"city": "Rochester NY", "state": "NY"},
        {"city": "Syracuse NY", "state": "NY"},
    ]
    df = pd.DataFrame(rows)
    props = propose_r5_compound_split(df)

    assert len(props) >= 1
    p = props[0]
    assert p.tier == "REVIEW"
    assert "city" in p.columns and "state" in p.columns

    # Verify changes
    city_changes = [c for c in p.changes if c.column == "city"]
    state_changes = [c for c in p.changes if c.column == "state"]

    assert any(c.old_value == "San Francisco CA" and c.new_value == "San Francisco" for c in city_changes)
    assert any(c.old_value == "" and c.new_value == "CA" for c in state_changes)


def test_r5_compound_split_safety_no_overwrite_conflict():
    # If state already has 'TX', do not overwrite with 'CA'
    rows = [
        {"city": "San Francisco CA", "state": "TX"},  # Conflict
        {"city": "San Jose CA", "state": ""},
        {"city": "Oakland CA", "state": "CA"},
        {"city": "Sacramento CA", "state": "CA"},
        {"city": "Fresno CA", "state": "CA"},
        {"city": "Los Angeles CA", "state": "CA"},
        {"city": "San Diego CA", "state": "CA"},
    ]
    df = pd.DataFrame(rows)
    props = propose_r5_compound_split(df)
    for p in props:
        for c in p.changes:
            if c.row == 0 and c.column == "state":
                pytest.fail("R5 should not overwrite conflicting non-blank column B")


def test_r6_dependency_repair_basic():
    # zip -> city functional dependency: 94101 -> 'San Francisco' (10 times), 1 typo 'San Francicso'
    rows = []
    for i in range(20):
        if i == 5:
            city_val = "San Francicso"  # typo
        elif i == 6:
            city_val = ""  # blank to fill
        else:
            city_val = "San Francisco"
        rows.append({"zip": "94101", "city": city_val})

    # Add second zip group
    for i in range(10):
        rows.append({"zip": "10001", "city": "New York"})

    df = pd.DataFrame(rows)
    props = propose_r6_dependency_repair(df)

    assert len(props) == 1
    p = props[0]
    assert p.tier == "REVIEW"
    assert p.columns == ["zip", "city"]

    # Verify typo repaired to 'San Francisco'
    typo_change = next(c for c in p.changes if c.row == 5)
    assert typo_change.old_value == "San Francicso"
    assert typo_change.new_value == "San Francisco"

    # Verify blank filled with 'San Francisco'
    blank_change = next(c for c in p.changes if c.row == 6)
    assert blank_change.old_value == ""
    assert blank_change.new_value == "San Francisco"


def test_r6_no_repair_when_low_confidence():
    # Low confidence: zip has split evenly 50/50 between two cities
    rows = []
    for i in range(20):
        city = "CityA" if i % 2 == 0 else "CityB"
        rows.append({"zip": "90001", "city": city})

    df = pd.DataFrame(rows)
    props = propose_r6_dependency_repair(df)
    assert props == []


def test_r8_exact_duplicates():
    data = {
        "id": ["1", "2", "3", "2", "4"],
        "name": ["A", "B", "C", "B", "E"]
    }
    df = pd.DataFrame(data)
    props = propose_r8_exact_duplicates(df)

    assert len(props) == 1
    p = props[0]
    assert p.kind == "R8_exact_duplicates"
    assert p.tier == "REVIEW"
    assert p.dropped_rows == [3]  # Row 3 is duplicate of Row 1


def test_phase2_engine_idempotency():
    rows = [
        {"zip": "94101", "city": "San Francisco CA", "state": "CA"},
        {"zip": "94101", "city": "San Francisco CA", "state": "CA"},
        {"zip": "94101", "city": "San Francicso", "state": "CA"},
        {"zip": "94101", "city": "San Francisco CA", "state": ""},
        {"zip": "94101", "city": "San Francisco CA", "state": "CA"},
        {"zip": "94101", "city": "San Francisco CA", "state": "CA"},
        {"zip": "94101", "city": "San Francisco CA", "state": "CA"},
    ]
    df = pd.DataFrame(rows)
    engine = CleaningEngine()

    props = engine.generate_proposals(df)
    # Approve all proposals
    cleaned_df, diffs = engine.apply(df, props)

    assert len(cleaned_df) < len(df)  # Duplicates were dropped
    assert len(diffs) > 0

    # Idempotency: re-running on cleaned output produces 0 proposals / 0 diffs
    props_pass2 = engine.generate_proposals(cleaned_df)
    cleaned_pass2, diffs_pass2 = engine.apply(cleaned_df, props_pass2)

    assert diffs_pass2 == []
    pd.testing.assert_frame_equal(cleaned_df, cleaned_pass2)
