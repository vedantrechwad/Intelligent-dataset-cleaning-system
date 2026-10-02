"""
src/intelligence/semantic_standardizer.py
AI-Powered Semantic Vocabulary Standardizer and Entity Disambiguator.

Architecture:
- Operates at the Column Vocabulary level (O(1) LLM calls), NEVER row-by-row (O(N)).
- Extracts unique categorical strings (5 to 120 unique items).
- Queries local Ollama to cluster synonyms, abbreviations, and domain variations into canonical labels.
- Returns a structured Proposal (Tier: REVIEW) with deterministic CellChange records.
- Fast vectorized execution in Pandas: df[col] = df[col].map(mapping).fillna(df[col]).
- Zero-hallucination guarantee: only maps explicitly reviewed values; untouched cells remain identical.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd

from cleaner.rules.base import Proposal, CellChange
from src.intelligence.ollama_client import OllamaClient
from src.utils.helpers import logger


def generate_semantic_vocabulary_proposal(
    df: pd.DataFrame,
    col: str,
    ollama_client: OllamaClient,
    touched_cells: Optional[Set[Tuple[int, str]]] = None,
    max_unique_values: int = 100,
    min_unique_values: int = 4
) -> Optional[Proposal]:
    """
    Standardize noisy categorical column vocabularies using local Ollama.
    1. Extracts unique values & frequencies.
    2. Sends full vocabulary in a single prompt to Ollama.
    3. Receives canonical mapping dictionary.
    4. Compares old vs new, builds CellChange records.
    5. Returns reviewable Proposal.
    """
    if touched_cells is None:
        touched_cells = set()

    if col not in df.columns:
        return None

    if not ollama_client.check_availability():
        return None

    series = df[col].dropna().astype(str)
    # Ignore empty or blank
    non_blank = series[series.str.strip() != ""]
    if len(non_blank) < 10:
        return None

    val_counts = non_blank.value_counts()
    n_unique = len(val_counts)

    if n_unique < min_unique_values or n_unique > max_unique_values:
        return None

    # Don't run on purely numeric or code-like columns
    if pd.to_numeric(non_blank, errors="coerce").notna().mean() > 0.5:
        return None

    vocab_with_counts = {str(k): int(v) for k, v in val_counts.items()}
    vocab_list = list(vocab_with_counts.keys())

    prompt = f"""You are an expert data taxonomist and master data management engine.
Your task is to standardize and harmonize the vocabulary of the categorical column "{col}".

Here are all {n_unique} unique values found in this column, along with their occurrence frequencies:
{json.dumps(vocab_with_counts, indent=2)}

Guidelines:
1. Identify abbreviations, alternate spellings, synonyms, or casing/punctuation variations that clearly refer to the same entity (e.g., 'Sr. Dev' -> 'Senior Developer', 'Elec Check' -> 'Electronic Check').
2. Choose the most complete, professional, and standard canonical form (prefer the more frequent standard name).
3. CRITICAL SAFETY GUARD: NEVER merge terms that represent distinct categories, opposites, or distinct entities (e.g., 'North' vs 'South', 'Male' vs 'Female', 'Yes' vs 'No', 'Pediatrics' vs 'Podiatry', 'DSL' vs 'Fiber Optic').
4. For any value that is already standard or has no clear duplicate, map it to itself.

Respond ONLY with a valid JSON object matching this schema:
{{
    "canonical_mapping": {{
        "variant_string_1": "Standard Form",
        "variant_string_2": "Standard Form"
    }},
    "reasoning": "Brief explanation of what was standardized and why."
}}
"""

    payload = {
        "model": ollama_client.model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    try:
        r = requests.post(f"{ollama_client.host}/api/generate", json=payload, timeout=ollama_client.timeout * 2)
        if r.status_code != 200:
            return None

        data = r.json()
        raw_resp = data.get("response", "{}")
        parsed = json.loads(raw_resp)
        mapping = parsed.get("canonical_mapping", {})
        reasoning = parsed.get("reasoning", f"AI Semantic standardization for column '{col}'.")

        if not isinstance(mapping, dict):
            return None

        # Filter out self-mappings
        real_mappings = {str(k): str(v) for k, v in mapping.items() if str(k) != str(v) and str(k) in vocab_with_counts}
        if not real_mappings:
            return None

        # Build CellChange records
        changes: List[CellChange] = []
        for idx, val in series.items():
            str_val = str(val)
            if str_val in real_mappings:
                cell_key = (int(idx), col)
                if cell_key not in touched_cells:
                    changes.append(CellChange(
                        row=int(idx),
                        column=col,
                        old_value=str_val,
                        new_value=real_mappings[str_val]
                    ))

        if not changes:
            return None

        prop_id = f"R9_semantic_vocab_{col}"
        summary_evidence = ", ".join([f"'{k}' -> '{v}'" for k, v in list(real_mappings.items())[:4]])
        if len(real_mappings) > 4:
            summary_evidence += f" (+{len(real_mappings) - 4} more)"

        return Proposal(
            id=prop_id,
            kind="R9_semantic_vocabulary",
            tier="REVIEW",
            columns=[col],
            description=f"AI Semantic Vocabulary Harmonization for '{col}': Standardize {len(real_mappings)} variant terms to canonical ontology.",
            evidence=f"{reasoning} Mappings: {summary_evidence}",
            changes=changes
        )

    except Exception as e:
        logger.warning(f"Semantic vocabulary standardization failed for {col}: {e}")
        return None


def semantic_tie_break(
    term_a: str,
    term_b: str,
    column_name: str,
    ollama_client: OllamaClient
) -> Tuple[bool, str]:
    """
    Arbiter for borderline near-miss categories in R7.
    Returns:
        (should_merge: bool, explanation: str)
    """
    if not ollama_client.check_availability():
        return False, "Ollama offline; conservative default applied (no merge)."

    prompt = f"""In a dataset with column "{column_name}", we have two similar values:
Value 1: "{term_a}"
Value 2: "{term_b}"

Are these two values:
A) The exact same real-world entity with a typo/spelling variation?
OR
B) Two distinct, separate categories or entities that should NOT be merged?

Respond ONLY with a JSON object:
{{
    "are_same_entity": true or false,
    "confidence": 0.95,
    "explanation": "1-sentence explanation"
}}
"""

    payload = {
        "model": ollama_client.model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0}
    }

    try:
        r = requests.post(f"{ollama_client.host}/api/generate", json=payload, timeout=10)
        if r.status_code == 200:
            parsed = json.loads(r.json().get("response", "{}"))
            same = bool(parsed.get("are_same_entity", False))
            expl = str(parsed.get("explanation", ""))
            return same, expl
    except Exception as e:
        logger.warning(f"Semantic tie-break failed: {e}")

    return False, "Fallback conservative: do not merge."
