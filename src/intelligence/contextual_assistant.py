"""
src/intelligence/contextual_assistant.py
AI-Powered Contextual Assistant for Reviewing Ambiguous Non-Numerical Data.
Explains anomalies (names, dates of birth, categories) and suggests ground-truth fixes using row context.
Includes graceful deterministic fallback when local Ollama is offline.
"""

import json
import re
from typing import Dict, Any, Optional
import pandas as pd
from src.intelligence.ollama_client import OllamaClient
from src.utils.helpers import logger


def explain_and_suggest_fix(
    row_dict: Dict[str, Any],
    target_col: str,
    target_val: Any,
    anomaly_desc: str,
    ollama_client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """
    Examines row context and provides an explanation and suggested fix for an ambiguous cell.
    Returns:
        {
            "explanation": str,
            "suggested_value": str or None,
            "confidence": float,
            "source": "ollama" | "heuristic"
        }
    """
    target_str = str(target_val) if target_val is not None else ""

    # 1. Deterministic Heuristic Analysis (Fast baseline)
    heuristic_fix = None
    heuristic_reason = None

    # Check Date of Birth / Year anomalies
    if re.search(r"birth|dob|born|year", target_col, re.IGNORECASE):
        # Look for century typo (e.g. 18xx instead of 19xx, or 20xx)
        m = re.search(r"\b(18\d{2})\b", target_str)
        if m:
            candidate = target_str.replace(m.group(1), f"19{m.group(1)[2:]}")
            heuristic_fix = candidate
            heuristic_reason = f"Identified likely century typo in '{target_col}' ({m.group(1)} -> 19{m.group(1)[2:]})."

    # Check name with digits/typos (e.g. OCR or leetspeak substitutions: 0->o, 1->i, 3->e)
    if not heuristic_fix and re.search(r"name|first|last|customer|patient", target_col, re.IGNORECASE):
        leet_map = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "8": "b", "@": "a", "$": "s"}
        decoded = target_str
        for num_char, letter in leet_map.items():
            decoded = decoded.replace(num_char, letter)
        cleaned_name = re.sub(r"[0-9_\-!@#$%^&*()]", "", decoded).strip()
        if cleaned_name and cleaned_name.lower() != target_str.lower():
            heuristic_fix = cleaned_name.title()
            heuristic_reason = f"Corrected OCR/typographical character substitutions in name: '{target_str}' -> '{heuristic_fix}'."

    # 2. Try Local Ollama LLM if available
    if ollama_client is not None and ollama_client.check_availability():
        try:
            # Filter context to non-empty fields (max 6 fields to keep prompt compact)
            compact_ctx = {
                k: v for k, v in row_dict.items()
                if v is not None and str(v).strip() != "" and k != target_col
            }
            subset_ctx = dict(list(compact_ctx.items())[:6])

            prompt = f"""You are an expert data curation assistant inspecting an anomaly in a tabular dataset.
Row context:
{json.dumps(subset_ctx, indent=2)}

Flagged Column: "{target_col}"
Current Value: "{target_str}"
Flagged Issue: "{anomaly_desc}"

Task:
1. Briefly explain in 1-2 plain-English sentences what is likely wrong, using the other columns for context.
2. If confident, provide the most likely intended ground-truth value. If unknown, output null for suggested_value.

Respond ONLY with a valid JSON object matching this schema:
{{
    "explanation": "concise explanation",
    "suggested_value": "corrected value or null"
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

            resp = ollama_client._post_generate(payload)
            if resp and "response" in resp:
                data = json.loads(resp["response"])
                sugg = data.get("suggested_value")
                expl = data.get("explanation", "").strip()
                if expl:
                    return {
                        "explanation": expl,
                        "suggested_value": sugg if sugg and str(sugg).lower() != "null" else heuristic_fix,
                        "confidence": 0.85 if sugg else 0.50,
                        "source": "ollama"
                    }
        except Exception as e:
            logger.warning(f"Ollama reasoning failed, falling back to heuristic: {e}")

    # Fallback response
    return {
        "explanation": heuristic_reason or f"Value '{target_str}' in column '{target_col}' flagged for verification ({anomaly_desc}).",
        "suggested_value": heuristic_fix,
        "confidence": 0.70 if heuristic_fix else 0.30,
        "source": "heuristic"
    }
