import json
import requests
from typing import Dict, Any, Optional, List
from src.utils.helpers import logger
from src.intelligence.ollama_client import OllamaClient

def impute_missing_value_with_llm(
    client: OllamaClient,
    row_dict: Dict[str, Any],
    target_col: str,
    col_stats: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Passes the context of an entire row to Ollama to intelligently 
    infer the most probable value for a missing cell.
    """
    if not client.check_availability():
        return None

    # Filter out the missing column from context
    context = {k: v for k, v in row_dict.items() if k != target_col and v is not None and str(v).strip() != ""}

    frequent_vals = col_stats.get("top_frequent_values", {})
    freq_list = list(frequent_vals.keys())[:10] if isinstance(frequent_vals, dict) else []

    prompt = f"""You are an expert data analyst and imputation engine.
Your goal is to infer a missing value in a dataset record based on the context of the other columns in the same row.

Missing Column Name: "{target_col}"
Common/Valid Values for this column in the dataset: {json.dumps(freq_list)}

Row Context (Other known values for this specific record):
{json.dumps(context, indent=2)}

Task:
Based on the Row Context, infer the most logical and probable value for "{target_col}".
If it is a categorical column, try to pick from the Common/Valid Values if one logically fits.
If it is numeric, estimate a reasonable numerical value based on correlations you know from general domain knowledge.

Respond ONLY with a valid JSON object strictly matching this schema:
{{
    "inferred_value": "The value you predict",
    "confidence": 0.85,
    "reason": "Brief 1-sentence explanation of why this value makes sense given the row context."
}}
"""

    payload = {
        "model": client.model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    try:
        r = requests.post(f"{client.host}/api/generate", json=payload, timeout=client.timeout)
        if r.status_code == 200:
            raw_response = r.json().get("response", "")
            parsed = json.loads(raw_response)
            
            if "inferred_value" in parsed and "confidence" in parsed:
                return {
                    "inferred_value": parsed.get("inferred_value"),
                    "confidence": float(max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))),
                    "reason": str(parsed.get("reason", "Inferred from LLM context."))
                }
    except Exception as e:
        logger.warning(f"Ollama imputation query failed: {e}")

    return None
