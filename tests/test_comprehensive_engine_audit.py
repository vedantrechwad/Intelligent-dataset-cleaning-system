"""
tests/test_comprehensive_engine_audit.py
Rigorous verification suite for:
1. Invariants (Byte-identity, Idempotency, Non-interference coordinate lock)
2. Standalone script fidelity via isolated subprocess execution
3. Edge cases, chaos datasets, custom/non-consecutive indexes, and semantic barriers
"""

import os
import sys
import tempfile
import subprocess
import pandas as pd
import pytest

from cleaner.engine import CleaningEngine
from cleaner.profile import profile_table
from cleaner.exporter import generate_cleaning_script
from cleaner.rules.r6_dependency_repair import propose_r6_dependency_repair
from cleaner.rules.r7_category_variants import propose_r7_category_variants


def test_invariant_untouched_cell_byte_identity():
    """
    Assert that every cell coordinate not explicitly targeted by an approved proposal
    is 100% byte-identical to the raw input.
    """
    df = pd.DataFrame({
        "id": ["001", "002", "003", "004", "005"],
        "city": ["New York", "N/A", "Chicago", "null", "Boston"],
        "state": ["NY", "NY", "IL", "IL", "MA"],
        "zip": ["10001", "10002", "60601", "60602", "02108"],
        "notes": ["Special char: äöü €", "Leading zero intact", "Keep exact", "Don't touch", "100% match"]
    })

    engine = CleaningEngine()
    prof = profile_table(df)
    proposals = engine.generate_proposals(df, prof)

    approved = [p for p in proposals if p.tier == "AUTO"]
    cleaned_df, diffs = engine.apply(df, approved)

    targeted_cells = set((d["row"], d["column"]) for d in diffs if d["column"] != "__ROW__")

    for r in df.index:
        for c in df.columns:
            if (r, c) not in targeted_cells:
                assert cleaned_df.at[r, c] == df.at[r, c], f"Untouched cell ({r}, {c}) mutated!"


def test_invariant_idempotency():
    """
    Assert that cleaning an already cleaned dataset produces zero additional changes:
    f(f(X)) == f(X)
    """
    df = pd.DataFrame({
        "code": ["  A1  ", "A1", "B2", "  B2 ", "N/A", ""],
        "amount": ["$1,200.50", "$1200.50", "300.00", "300.00", "50.00", "0.00"],
        "category": ["Retail", "retail", "RETAIL", "Wholesale", "Wholesale", "Retail"]
    })

    engine = CleaningEngine()
    prof1 = profile_table(df)
    props1 = engine.generate_proposals(df, prof1)
    approved1 = [p for p in props1 if p.tier in ("AUTO", "REVIEW")]
    cleaned_first, diffs1 = engine.apply(df, approved1)

    # Second pass
    prof2 = profile_table(cleaned_first)
    props2 = engine.generate_proposals(cleaned_first, prof2)
    approved2 = [p for p in props2 if p.tier in ("AUTO", "REVIEW")]
    cleaned_second, diffs2 = engine.apply(cleaned_first, approved2)

    assert len(diffs2) == 0, f"Engine is not idempotent! Second pass generated {len(diffs2)} additional changes."
    assert cleaned_second.equals(cleaned_first)


def test_invariant_coordinate_non_interference_lock():
    """
    Assert that no two proposals target the same cell coordinate (r, c).
    """
    df = pd.DataFrame({
        "val": ["  N/A  ", "  $1,000  ", "null", "12.0 oz.", "  Chicago IL  "]
    })

    engine = CleaningEngine()
    proposals = engine.generate_proposals(df, profile_table(df))

    seen_coords = set()
    for prop in proposals:
        for ch in prop.changes:
            coord = (ch.row, ch.column)
            assert coord not in seen_coords, f"Coordinate collision on {coord} by {prop.id}!"
            seen_coords.add(coord)


def test_custom_and_fragmented_index_resilience():
    """
    Verify engine and R6 functional dependency repair when DataFrame has
    non-consecutive integer indices and custom string indices.
    """
    # 1. Non-consecutive integer index (e.g. after row drops)
    df_gapped = pd.DataFrame(
        {
            "zip": ["90210", "90210", "90210", "10001", "10001", "10001", "60601", "60601"],
            "city": ["Beverly Hills", "", "Beverly Hills", "New York", "", "New York", "Chicago", "Chicago"]
        },
        index=[10, 25, 42, 100, 150, 200, 310, 420]
    )

    r6_props = propose_r6_dependency_repair(df_gapped)
    assert len(r6_props) > 0
    engine = CleaningEngine()
    cleaned_gapped, diffs = engine.apply(df_gapped, r6_props)

    assert cleaned_gapped.loc[25, "city"] == "Beverly Hills"
    assert cleaned_gapped.loc[150, "city"] == "New York"

    # 2. String index labels
    df_str_idx = pd.DataFrame(
        {
            "zip": ["90210", "90210", "90210", "10001", "10001", "10001", "60601", "60601"],
            "city": ["Beverly Hills", "", "Beverly Hills", "New York", "", "New York", "Chicago", "Chicago"]
        },
        index=["row_A", "row_B", "row_C", "row_D", "row_E", "row_F", "row_G", "row_H"]
    )

    r6_props_str = propose_r6_dependency_repair(df_str_idx)
    assert len(r6_props_str) > 0
    cleaned_str, diffs_str = engine.apply(df_str_idx, r6_props_str)
    assert cleaned_str.loc["row_B", "city"] == "Beverly Hills"
    assert cleaned_str.loc["row_E", "city"] == "New York"


def test_exporter_fidelity_against_engine_output():
    """
    Verify that the generated standalone Python script runs independently
    in a child Python subprocess and produces an output CSV that matches
    engine.apply() 100% cell-by-cell.
    """
    raw_df = pd.DataFrame({
        "order_id": ["001", "002", "003", "004", "005", "006"],
        "zip": ["90210", "90210", "90210", "10001", "10001", "10001"],
        "city": ["Beverly Hills", "", "Beverly Hills", "New York", "", "New York"],
        "price": ["$1,200.00", "$1,200.00", "$1,200.00", "$450.00", "$450.00", "$450.00"],
        "status": ["pending", "Pending", "PENDING", "delivered", "delivered", "delivered"]
    })

    engine = CleaningEngine()
    prof = profile_table(raw_df)
    proposals = engine.generate_proposals(raw_df, prof)
    approved = [p for p in proposals if p.tier in ("AUTO", "REVIEW")]

    # Engine result
    engine_cleaned, _ = engine.apply(raw_df, approved)

    # Generate standalone script
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "dirty_input.csv")
        output_csv = os.path.join(tmpdir, "script_output.csv")
        script_py = os.path.join(tmpdir, "run_clean.py")

        raw_df.to_csv(input_csv, index=False, encoding="utf-8")

        script_code = generate_cleaning_script(approved, original_filename="dirty_input.csv")
        with open(script_py, "w", encoding="utf-8") as fp:
            fp.write(script_code)

        # Run standalone script via python subprocess
        res = subprocess.run(
            [sys.executable, script_py, input_csv, "-o", output_csv],
            capture_output=True,
            text=True
        )
        assert res.returncode == 0, f"Standalone script failed with error:\n{res.stderr}"

        # Load script output strictly as strings
        script_cleaned = pd.read_csv(output_csv, dtype=str, keep_default_na=False)

        # Assert 100% equivalence
        assert script_cleaned.shape == engine_cleaned.shape
        assert script_cleaned.equals(engine_cleaned), (
            f"Standalone script output differs from engine output!\n"
            f"Engine:\n{engine_cleaned}\nScript:\n{script_cleaned}"
        )


def test_semantic_opposite_barriers_in_r7():
    """
    Assert that categorical variant clustering strictly forbids merging antonyms
    (male/female, yes/no, pass/fail) regardless of edit distance.
    """
    df = pd.DataFrame({
        "gender": ["male"] * 98 + ["female"] * 2,
        "decision": ["pass"] * 98 + ["fail"] * 2,
        "flag": ["true"] * 98 + ["false"] * 2
    })

    props = propose_r7_category_variants(df)
    assert len(props) == 0, "R7 incorrectly proposed merging semantic opposites!"


def test_chaos_empty_and_single_cell_tables():
    """
    Assert engine resilience on extreme degenerate tabular structures:
    - 0-row empty DataFrame
    - 1-row DataFrame
    - 1-column 100% blank DataFrame
    """
    engine = CleaningEngine()

    # 1. Zero rows
    df_empty = pd.DataFrame(columns=["a", "b", "c"])
    props_empty = engine.generate_proposals(df_empty)
    cleaned_empty, diffs_empty = engine.apply(df_empty, props_empty)
    assert len(cleaned_empty) == 0

    # 2. Single row
    df_single = pd.DataFrame([{"a": "hello", "b": "10.5"}])
    props_single = engine.generate_proposals(df_single)
    cleaned_single, diffs_single = engine.apply(df_single, props_single)
    assert len(cleaned_single) == 1

    # 3. 100% blank column
    df_blank = pd.DataFrame({"col1": ["", "", ""], "col2": ["1", "2", "3"]})
    props_blank = engine.generate_proposals(df_blank)
    cleaned_blank, diffs_blank = engine.apply(df_blank, props_blank)
    assert cleaned_blank.shape == df_blank.shape
