"""
cleaner/rules/r7_category_variants.py
R7 Category Variants:
Clusters near-miss strings in low-cardinality columns using Levenshtein distance (threshold <= 2)
or casing/punctuation/whitespace normalization.

Guards:
- Free-text categories only: median length > 3, not code-like, not numeric-with-unit.
- Variant must be rare (<= 2% of column) and >= 10x rarer than target.
- Differs by <= 2 edit distance (with length >= 6), or differs only by casing/punctuation/whitespace.
- Never merge if differing by a whole word (e.g. 'North Carolina' vs 'South Carolina').
- Tier: REVIEW with per-cluster confirmation, showing counts on both sides.
"""

import re
from typing import List, Set, Dict, Any, Optional, Tuple
import pandas as pd
from cleaner.rules.base import Proposal, CellChange


# Common opposite or semantically distinct categorical words that must never merge
OPPOSITE_OR_DISTINCT_WORDS = {
    "north", "south", "east", "west", "upper", "lower", "male", "female",
    "yes", "no", "true", "false", "on", "off", "in", "out", "city", "county",
    "state", "first", "second", "third", "last", "new", "old", "big", "small",
    "high", "low", "left", "right", "red", "blue", "green", "black", "white",
    "pos", "neg", "positive", "negative", "pass", "fail", "am", "pm"
}

NUMERIC_WITH_UNIT_RE = re.compile(r"^[-+]?\d+(\.\d+)?\s*[a-zA-Z%]+$")
CODE_LIKE_RE = re.compile(r"^[A-Z0-9_\-./]{1,5}$")


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(s2)]


def normalize_punctuation_space(s: str) -> str:
    """Lowercase and normalize whitespace and punctuation for comparison."""
    return re.sub(r"[\s\-_.,/]+", " ", s.lower()).strip()


def extract_word_tokens(s: str) -> List[str]:
    """Extract individual alphanumeric word tokens in lowercase."""
    return [w.lower() for w in re.findall(r"\b\w+\b", s)]


def is_valid_near_miss(variant: str, target: str) -> Tuple[bool, str, int]:
    """
    Check if variant is a valid near-miss of target.
    Returns:
        (is_valid, reason, distance)
    """
    # 1. Differs only by casing/punctuation/whitespace
    norm_v = normalize_punctuation_space(variant)
    norm_t = normalize_punctuation_space(target)
    if norm_v == norm_t and variant != target:
        dist = levenshtein_distance(variant.lower(), target.lower())
        return True, "casing_punctuation_whitespace", dist

    # 2. Levenshtein distance <= 2 with length >= 6
    if len(variant) < 6 or len(target) < 6:
        return False, "length_too_short (<6)", -1

    dist = levenshtein_distance(variant.lower(), target.lower())
    if dist > 2 or dist == 0:
        return False, f"edit_distance_{dist}", dist

    # 3. Whole-word guard: Check word-level differences
    words_v = extract_word_tokens(variant)
    words_t = extract_word_tokens(target)

    # If word counts differ:
    if len(words_v) != len(words_t):
        # Allow only if space insertion/deletion (e.g. 'Wal Mart' vs 'Walmart')
        if variant.replace(" ", "").lower() == target.replace(" ", "").lower():
            return True, "spacing_difference", dist
        return False, "word_count_mismatch", dist

    # Word counts are equal: identify differing word tokens
    differing_words = [(wv, wt) for wv, wt in zip(words_v, words_t) if wv != wt]
    if len(differing_words) != 1:
        return False, "multiple_words_differ", dist

    wv, wt = differing_words[0]

    # Guard against opposite or semantically distinct categorical words
    if wv in OPPOSITE_OR_DISTINCT_WORDS or wt in OPPOSITE_OR_DISTINCT_WORDS:
        return False, f"semantic_distinction_{wv}_vs_{wt}", dist

    # Guard against replacing a whole short word (e.g. 4-letter distinct words)
    if min(len(wv), len(wt)) < 5:
        return False, f"short_word_differ_{wv}_vs_{wt}", dist

    # The differing word must itself have edit distance <= 2
    word_dist = levenshtein_distance(wv, wt)
    if word_dist > 2:
        return False, f"differing_word_distance_{word_dist}", dist

    return True, f"edit_distance_{dist}", dist


def propose_r7_category_variants(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    touched_cells: Optional[Set[Tuple[int, str]]] = None
) -> List[Proposal]:
    """
    Propose R7 categorical near-miss consolidation.
    Always Tier: REVIEW.
    """
    if touched_cells is None:
        touched_cells = set()

    proposals: List[Proposal] = []
    n_rows = len(df)
    if n_rows < 10:
        return proposals

    for col in df.columns:
        series = df[col]
        # Only evaluate non-empty cells
        valid_items = [(idx, str(val).strip()) for idx, val in series.items() if str(val).strip() != ""]
        n_valid = len(valid_items)
        if n_valid < 10:
            continue

        # Count frequencies
        freq: Dict[str, int] = {}
        for _, val in valid_items:
            freq[val] = freq.get(val, 0) + 1

        n_unique = len(freq)
        # Must be low-to-moderate cardinality category
        if n_unique <= 1 or n_unique / n_valid > 0.35 or n_unique > 200:
            continue

        # Check median string length > 3
        lengths = [len(val) for val in freq.keys()]
        median_len = sorted(lengths)[len(lengths) // 2]
        if median_len <= 3:
            continue

        # Guard: not code-like (e.g. state abbreviations or short alphanumeric codes)
        code_like_count = sum(1 for val in freq.keys() if CODE_LIKE_RE.match(val))
        if code_like_count / n_unique > 0.70:
            continue

        # Guard: not numeric with unit
        unit_count = sum(1 for val in freq.keys() if NUMERIC_WITH_UNIT_RE.match(val))
        if unit_count / n_unique > 0.50:
            continue

        # Identify candidate targets and rare variants
        # Variant must be rare: frequency <= 2% of valid items, or <= 2 count if n_valid < 100
        # Target must be frequent: frequency >= 10x variant, and >= 3
        rare_variants = [v for v, c in freq.items() if (c / n_valid <= 0.02 or (n_valid < 100 and c <= 2))]
        if not rare_variants:
            continue

        # Candidate targets
        targets = [t for t, c in freq.items() if c >= 3 and t not in rare_variants]
        if not targets:
            continue

        # For each rare variant, find best matching target
        for variant in rare_variants:
            c_v = freq[variant]
            best_target = None
            best_dist = 999
            best_t_count = -1

            for target in targets:
                c_t = freq[target]
                if c_t < 10 * c_v:
                    continue

                valid, reason, dist = is_valid_near_miss(variant, target)
                if valid:
                    # Prefer higher frequency target, tie break by lower edit distance
                    if c_t > best_t_count or (c_t == best_t_count and dist < best_dist):
                        best_target = target
                        best_dist = dist
                        best_t_count = c_t

            if best_target is not None:
                # Find all occurrences of this variant in column not already touched
                changes: List[CellChange] = []
                for idx, val in valid_items:
                    if val == variant and (idx, col) not in touched_cells:
                        changes.append(CellChange(
                            row=int(idx),
                            column=col,
                            old_value=val,
                            new_value=best_target
                        ))

                if changes:
                    slug = re.sub(r"[^a-zA-Z0-9_]", "_", variant[:15]).strip("_")
                    proposals.append(Proposal(
                        id=f"R7_category_{col}_{slug}",
                        kind="R7_category_variants",
                        tier="REVIEW",
                        columns=[col],
                        description=f"Consolidate near-miss variant '{variant}' ({c_v} obs) into '{best_target}' ({best_t_count} obs) in column '{col}'",
                        evidence=(
                            f"Variant '{variant}' occurs {c_v} times (<= 2% share), while dominant category "
                            f"'{best_target}' occurs {best_t_count} times (>={best_t_count // max(1, c_v)}x ratio). "
                            f"Edit distance: {best_dist}. REVIEW tier: requires human confirmation."
                        ),
                        changes=changes
                    ))

    return proposals
