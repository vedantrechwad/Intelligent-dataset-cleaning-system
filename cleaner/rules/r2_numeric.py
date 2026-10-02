"""
cleaner/rules/r2_numeric.py
R2 Numeric canonicalization:
For columns where >=60% of non-missing values parse after stripping a single dominant currency symbol,
%, thousands separator, or unit token (case/punctuation variants), convert the minority formats to the
dominant canonical form. Never change the numeric value or rescale (0.05% -> 0.05, not 5);
ambiguous scale (% present in some values, absent in others) is a REVIEW proposal showing both interpretations.
Descriptor text that will be dropped is shown in the preview.
"""

import re
from typing import List, Set, Dict, Any, Optional, Tuple
import pandas as pd
from cleaner.rules.base import Proposal, CellChange


# Common physical & digital units (lowercase)
STANDARD_UNITS: Set[str] = {
    "oz", "oz.", "ounce", "ounces", "fl oz", "fl. oz.", "fl oz.",
    "lb", "lbs", "pound", "pounds", "kg", "g", "mg", "ml", "l",
    "liter", "liters", "m", "cm", "mm", "km", "mi", "mile", "miles",
    "ft", "in", "inch", "inches", "yd", "sec", "s", "min", "mins",
    "hr", "hrs", "day", "days", "yr", "yrs", "kb", "mb", "gb", "tb",
    "hz", "khz", "mhz", "ghz", "v", "mv", "kv", "w", "kw", "mw",
    "deg", "°c", "°f"
}

CURRENCY_SYMBOLS: Set[str] = {
    "$", "€", "£", "¥", "₹", "eur", "usd", "gbp", "cad", "aud", "chf", "jpy", "cny", "inr"
}

NUM_PATTERN = re.compile(
    r"^\s*([^\d\-+]*?)\s*([-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[-+]?[0-9]+(?:\.[0-9]+)?)\s*(.*?)\s*$"
)


def is_unit_suffix(suffix: str) -> bool:
    """Check if suffix is or starts with a recognized standard unit or %."""
    s_low = suffix.strip().lower()
    if not s_low:
        return True
    if s_low == "%" or "%" in s_low:
        return True
    for u in STANDARD_UNITS:
        if s_low == u or s_low.startswith(u + " ") or s_low.startswith(u + "."):
            return True
    return False


def parse_numeric_cell(val: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a cell into (prefix, number_str, suffix).
    Must parse after stripping a currency symbol, %, separator, or standard unit token.
    Returns None if not parseable as numeric with recognized symbols/units.
    """
    if not val or not val.strip():
        return None
    m = NUM_PATTERN.match(val)
    if not m:
        return None
    prefix = m.group(1).strip()
    num_str = m.group(2).replace(",", "")
    suffix = m.group(3).strip()

    prefix_low = prefix.lower()
    if prefix_low and prefix_low not in CURRENCY_SYMBOLS:
        return None

    if not is_unit_suffix(suffix):
        return None

    return prefix, num_str, suffix


def propose_r2_numeric(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    touched_cells: Optional[Set[tuple]] = None
) -> List[Proposal]:
    """
    Propose R2 numeric canonicalizations.
    """
    if touched_cells is None:
        touched_cells = set()

    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows == 0:
        return proposals

    for col in df.columns:
        series = df[col]
        non_missing = [(idx, val) for idx, val in series.items() if val != "" and (idx, col) not in touched_cells]
        if len(non_missing) < 5:
            continue

        parsed_cells = []
        for idx, val in non_missing:
            res = parse_numeric_cell(val)
            if res:
                parsed_cells.append((idx, val, res[0], res[1], res[2]))

        parse_rate = len(parsed_cells) / len(non_missing)
        if parse_rate < 0.60:
            continue

        # Analyze percentage formats
        has_pct = [c for c in parsed_cells if c[4] == "%" or "%" in c[4]]
        no_pct = [c for c in parsed_cells if "%" not in c[4]]

        # Case 1: Ambiguous scale - percentage present in some, absent in others
        if len(has_pct) > 0 and len(no_pct) > 0:
            pct_changes = []
            for idx, val, prefix, num_str, suffix in has_pct:
                pct_changes.append(
                    CellChange(row=int(idx), column=col, old_value=val, new_value=num_str)
                )

            count_pct = len(has_pct)
            count_plain = len(no_pct)
            proposals.append(Proposal(
                id=f"R2_numeric_scale_{col}",
                kind="R2_numeric",
                tier="REVIEW",
                columns=[col],
                description=f"Resolve ambiguous percentage scale in column '{col}' by canonicalizing to plain numeric format",
                evidence=(
                    f"{count_pct} values ({count_pct/len(parsed_cells)*100:.1f}%) have '%', "
                    f"while {count_plain} values ({count_plain/len(parsed_cells)*100:.1f}%) are plain numbers. "
                    f"Interpretation 1 (proposed): Strip '%' preserving numeric value (e.g. 0.05% -> 0.05). "
                    f"Interpretation 2: Rescale by 100 (e.g. 5% -> 0.05)."
                ),
                changes=pct_changes
            ))
            continue

        # Check if a single non-empty unit suffix is overwhelmingly dominant (>= 85% of parsed cells)
        suffix_counts: Dict[str, int] = {}
        for _, _, _, _, sfx in parsed_cells:
            s_clean = sfx.strip().lower()
            if s_clean:
                suffix_counts[s_clean] = suffix_counts.get(s_clean, 0) + 1

        # If 100% of numeric values already have '%', that IS the dominant format. Do not strip %.
        if len(has_pct) > 0 and len(no_pct) == 0:
            continue

        # If 100% of numeric values already share the exact same non-empty suffix, that IS the dominant format.
        if len(suffix_counts) == 1 and list(suffix_counts.values())[0] == len(parsed_cells):
            continue

        # Case 2: Unit tokens, currency symbols, and descriptor text
        auto_changes: List[CellChange] = []
        review_changes: List[CellChange] = []
        dropped_text_samples: Set[str] = set()

        for idx, val, prefix, num_str, suffix in parsed_cells:
            if val == num_str:
                continue

            prefix_low = prefix.lower().strip()
            is_valid_prefix = prefix_low == "" or prefix_low in CURRENCY_SYMBOLS

            suffix_clean = suffix.strip()
            suffix_low = suffix_clean.lower()

            if not is_valid_prefix and prefix_low != "":
                review_changes.append(
                    CellChange(row=int(idx), column=col, old_value=val, new_value=num_str)
                )
                dropped_text_samples.add(prefix)
                continue

            if suffix_low == "":
                auto_changes.append(
                    CellChange(row=int(idx), column=col, old_value=val, new_value=num_str)
                )
            elif suffix_low in STANDARD_UNITS:
                auto_changes.append(
                    CellChange(row=int(idx), column=col, old_value=val, new_value=num_str)
                )
            else:
                has_unit_prefix = False
                extra_text = suffix_clean
                for u in sorted(STANDARD_UNITS, key=len, reverse=True):
                    if suffix_low.startswith(u):
                        has_unit_prefix = True
                        extra_text = suffix_clean[len(u):].strip(" .,-")
                        break

                if has_unit_prefix and not extra_text:
                    auto_changes.append(
                        CellChange(row=int(idx), column=col, old_value=val, new_value=num_str)
                    )
                else:
                    review_changes.append(
                        CellChange(row=int(idx), column=col, old_value=val, new_value=num_str)
                    )
                    if extra_text:
                        dropped_text_samples.add(extra_text)
                    else:
                        dropped_text_samples.add(suffix_clean)

        if auto_changes:
            count = len(auto_changes)
            proposals.append(Proposal(
                id=f"R2_numeric_canonical_{col}",
                kind="R2_numeric",
                tier="AUTO",
                columns=[col],
                description=f"Canonicalize {count} numeric values by stripping currency/unit symbols in column '{col}'",
                evidence=f"{len(parsed_cells)} of {len(non_missing)} values ({parse_rate*100:.1f}%) parse as numeric. Normalized {count} values with standard unit/currency symbols.",
                changes=auto_changes
            ))

        if review_changes:
            count = len(review_changes)
            dropped_desc = ", ".join([f"'{t}'" for t in list(dropped_text_samples)[:5]])
            proposals.append(Proposal(
                id=f"R2_numeric_descriptor_{col}",
                kind="R2_numeric",
                tier="REVIEW",
                columns=[col],
                description=f"Strip descriptor text to canonicalize {count} numeric values in column '{col}'",
                evidence=f"Dropping descriptor text in {count} values. Sample dropped text: [{dropped_desc}]. Preview shows before and after.",
                changes=review_changes
            ))

    return proposals
