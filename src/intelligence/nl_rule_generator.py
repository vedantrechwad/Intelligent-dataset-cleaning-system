"""
src/intelligence/nl_rule_generator.py
Natural Language Invariant Rule Generator.
Translates plain-English business rules (e.g. 'Discount must be between 0 and 50%')
into executable Pandas expressions for validation and human review.
"""

import json
import re
from typing import Dict, Any, Optional, List
import pandas as pd
from src.intelligence.ollama_client import OllamaClient
from src.utils.helpers import logger


def generate_rule_from_natural_language(
    df: pd.DataFrame,
    nl_prompt: str,
    ollama_client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """
    Translates a plain-English rule prompt into an executable filter on the DataFrame.
    Returns:
        {
            "status": "success" | "error",
            "rule_name": str,
            "description": str,
            "pandas_query": str,
            "violating_row_indices": List[int],
            "n_violations": int,
            "explanation": str
        }
    """
    cols = list(df.columns)
    schema_info = {col: str(df[col].dtype) for col in cols}

    # 1. Try Ollama LLM
    if ollama_client is not None and ollama_client.check_availability():
        try:
            prompt = f"""You are a senior data engineer.
Translate the following plain-English business rule into a safe, valid Pandas query expression that finds VIOLATING rows.

Table Columns & Types:
{json.dumps(schema_info, indent=2)}

Business Rule: "{nl_prompt}"

Guidelines:
1. Provide a Pandas expression for df.query(...) that evaluates to True for rows that VIOLATE the rule.
2. Use backticks around column names with spaces (e.g. `Total Spent` < 0).
3. Only use existing column names from the provided schema.

Respond ONLY with a valid JSON object matching this schema:
{{
    "rule_name": "Short Name",
    "description": "What this rule enforces",
    "pandas_query": "valid query string for df.query(...)",
    "explanation": "Why this query identifies violations"
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
                query = data.get("pandas_query", "").strip()
                if query:
                    try:
                        viol_df = df.query(query)
                        viol_indices = [int(i) for i in viol_df.index]
                        return {
                            "status": "success",
                            "rule_name": data.get("rule_name", "Custom Rule"),
                            "description": data.get("description", nl_prompt),
                            "pandas_query": query,
                            "violating_row_indices": viol_indices,
                            "n_violations": len(viol_indices),
                            "explanation": data.get("explanation", ""),
                            "source": "ollama"
                        }
                    except Exception as q_err:
                        logger.warning(f"Ollama generated query failed to execute: {q_err}")
        except Exception as e:
            logger.warning(f"Ollama rule generation error: {e}")

    # 2. Heuristic fallback for common patterns
    query = None
    rule_name = "Custom Filter"
    lower_prompt = nl_prompt.lower()

    # Pattern: positive or >= 0
    for col in cols:
        col_clean = col.strip("`")
        if col_clean.lower() in lower_prompt:
            if "positive" in lower_prompt or "greater than 0" in lower_prompt or ">= 0" in lower_prompt:
                query = f"`{col_clean}` < 0"
                rule_name = f"{col_clean} Non-Negative"
                break
            elif "not null" in lower_prompt or "not empty" in lower_prompt:
                query = f"`{col_clean}`.isna() or `{col_clean}` == ''"
                rule_name = f"{col_clean} Required"
                break

    if query:
        try:
            viol_df = df.query(query)
            viol_indices = [int(i) for i in viol_df.index]
            return {
                "status": "success",
                "rule_name": rule_name,
                "description": nl_prompt,
                "pandas_query": query,
                "violating_row_indices": viol_indices,
                "n_violations": len(viol_indices),
                "explanation": f"Evaluated fallback query: {query}",
                "source": "heuristic"
            }
        except Exception as e:
            pass

    return {
        "status": "error",
        "error": f"Could not compile natural language prompt '{nl_prompt}' into an executable rule on these columns.",
        "columns_available": cols
    }
