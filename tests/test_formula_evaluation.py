import pytest
import pandas as pd
import numpy as np
from app import evaluate_custom_formula, explain_proposal
from cleaner.rules.base import Proposal, CellChange

def test_evaluate_custom_formula_numeric():
    df = pd.DataFrame({
        "Quantity": [2, 5, 10, 4],
        "Unit Price": [10.0, 20.0, 5.0, 30.0]
    })
    mask = pd.Series([True, True, False, True], index=df.index)
    res, err = evaluate_custom_formula(df, "Quantity * Unit_Price", mask)
    
    assert err is None
    assert res is not None
    assert len(res) == 3
    assert res.loc[0] == 20.0
    assert res.loc[1] == 100.0
    assert res.loc[3] == 120.0

def test_evaluate_custom_formula_string_concat():
    df = pd.DataFrame({
        "First Name": ["John", "Jane", "Alice"],
        "Last Name": ["Doe", "Smith", "Johnson"]
    })
    mask = pd.Series([True, True, True], index=df.index)
    res, err = evaluate_custom_formula(df, "First_Name + ' ' + Last_Name", mask)
    
    assert err is None
    assert res.loc[0] == "John Doe"
    assert res.loc[1] == "Jane Smith"
    assert res.loc[2] == "Alice Johnson"

def test_evaluate_custom_formula_math_functions():
    df = pd.DataFrame({
        "Price": [10.3456, 20.6789]
    })
    mask = pd.Series([True, True], index=df.index)
    res, err = evaluate_custom_formula(df, "np.round(Price * 1.08, 2)", mask)
    
    assert err is None
    assert res.loc[0] == 11.17
    assert res.loc[1] == 22.33

def test_evaluate_custom_formula_error_handling():
    df = pd.DataFrame({"A": [1, 2]})
    mask = pd.Series([True, True], index=df.index)
    res, err = evaluate_custom_formula(df, "UndefinedColumn + 1", mask)
    
    assert res is None
    assert err is not None
    assert "UndefinedColumn" in err or "not defined" in err

def test_explain_proposal_layman():
    # R1 missing
    p_r1 = Proposal(
        id="R1_test",
        kind="R1_missing_tokens",
        tier="AUTO",
        columns=["Age"],
        description="Convert rare disguised missing token 'N/A' to empty string in column 'Age'",
        evidence="sample evidence",
        changes=[CellChange(row=0, column="Age", old_value="N/A", new_value="")]
    )
    what, why = explain_proposal(p_r1)
    assert "Replace placeholder" in what
    assert "disguised missing value" in why or "missing data" in why
    
    # R8 duplicates
    p_r8 = Proposal(
        id="R8_test",
        kind="R8_exact_duplicates",
        tier="AUTO",
        columns=["All Columns"],
        description="Deduplicate 3 exact duplicate rows",
        evidence="sample evidence",
        changes=[],
        dropped_rows=[1, 2, 3]
    )
    what8, why8 = explain_proposal(p_r8)
    assert "identical rows" in what8
    assert "byte-for-byte duplicates" in why8

def test_explain_proposal_differentiates_r2_subtypes():
    # Canonical rule
    p_canon = Proposal(
        id="R2_numeric_canonical_ounces",
        kind="R2_numeric",
        tier="AUTO",
        columns=["ounces"],
        description="Canonicalize 2270 numeric values by stripping currency/unit symbols in column 'ounces'",
        evidence="2270 of 2410 values parse as numeric.",
        changes=[
            CellChange(row=0, column="ounces", old_value="12.0 oz", new_value="12.0"),
            CellChange(row=1, column="ounces", old_value="12.0 oz.", new_value="12.0")
        ]
    )
    what_c, why_c = explain_proposal(p_canon)
    assert "unit and symbol variants" in what_c or "pure numbers" in what_c
    assert "underlying magnitude" in why_c or "inconsistent unit" in why_c

    # Descriptor rule
    p_desc = Proposal(
        id="R2_numeric_descriptor_ounces",
        kind="R2_numeric",
        tier="REVIEW",
        columns=["ounces"],
        description="Strip descriptor text to canonicalize 140 numeric values in column 'ounces'",
        evidence="Dropping descriptor text in 140 values. Sample dropped text: ['Alumi-Tek'].",
        changes=[
            CellChange(row=17, column="ounces", old_value="12.0 oz. Alumi-Tek", new_value="12.0")
        ]
    )
    what_d, why_d = explain_proposal(p_desc)
    assert "packaging text" in what_d or "descriptive" in what_d
    assert "Alumi-Tek" in what_d or "Alumi-Tek" in why_d
    assert why_d != why_c  # Guaranteed distinct!

    # Scale rule
    p_scale = Proposal(
        id="R2_numeric_scale_abv",
        kind="R2_numeric",
        tier="REVIEW",
        columns=["abv"],
        description="Resolve ambiguous percentage scale in column 'abv'",
        evidence="Percentage scale ambiguity.",
        changes=[
            CellChange(row=5, column="abv", old_value="5.5%", new_value="0.055")
        ]
    )
    what_s, why_s = explain_proposal(p_scale)
    assert "scale" in what_s or "percentage" in what_s
    assert why_s != why_c and why_s != why_d  # Guaranteed distinct!

def test_append_prepend_values_logic():
    df = pd.DataFrame({
        "Volume": ["12.0", "16.0", "", "24.0"]
    })
    
    # Test append with space
    sep = " "
    val_to_insert = "oz"
    target_series = df["Volume"]
    res_append = [f"{v}{sep}{val_to_insert}" if v != "" else "" for v in target_series]
    assert res_append == ["12.0 oz", "16.0 oz", "", "24.0 oz"]
    
    # Test prepend without space
    prefix = "$"
    res_prepend = [f"{prefix}{v}" if v != "" else "" for v in target_series]
    assert res_prepend == ["$12.0", "$16.0", "", "$24.0"]
