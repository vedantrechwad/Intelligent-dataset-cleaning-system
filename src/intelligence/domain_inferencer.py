import json
from typing import Dict, Any
import pandas as pd
from src.intelligence.ollama_client import OllamaClient
from src.utils.helpers import logger
import requests

def infer_dynamic_domain_rules(df: pd.DataFrame, schema_info: Dict[str, Any], ollama_client: OllamaClient) -> Dict[str, Any]:
    """
    Uses Ollama to infer logical domain constraints (e.g. min, max)
    based on the column names, inferred types, and a small sample of values.
    Returns a dictionary of domain rules that can override the static config.
    """
    if not ollama_client.check_availability():
        logger.info("Ollama is offline; skipping dynamic domain inference.")
        return {}

    logger.info("Querying Ollama for dynamic domain inference...")
    
    # Construct a concise representation of the schema for the LLM
    context_schema = {}
    for col, info in schema_info.items():
        if info.get("logical_type") in ["numeric", "datetime"]:
            context_schema[col] = {
                "type": info.get("logical_type"),
                "hint": info.get("semantic_hint"),
                "sample": info.get("sample_values", [])[:3]
            }

    if not context_schema:
        return {}

    prompt = f"""You are a data quality expert. Analyze this dataset schema and infer logical domain rules for any columns where common sense dictates constraints.
    
    Schema Info:
    {json.dumps(context_schema, indent=2)}

    Determine which columns have obvious boundaries. For example:
    - An 'age' column should have min: 0, max: 120.
    - A 'price' or 'salary' should have min: 0.
    - A 'percentage' or 'discount' should be between 0 and 100.
    - A 'weight' or 'height' cannot be negative.
    
    Only provide rules for numeric columns where bounds are logically obvious. Do not invent rules for arbitrary IDs.

    Respond ONLY with a valid JSON object matching this structure (use column names exactly as keys):
    {{
        "column_name_1": {{"min": 0, "max": 100}},
        "column_name_2": {{"min": 0}}
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
        if r.status_code == 200:
            raw_response = r.json().get("response", "{}")
            parsed = json.loads(raw_response)
            logger.info(f"Dynamically inferred domain rules: {parsed}")
            return parsed
    except Exception as e:
        logger.warning(f"Domain inference query failed: {e}")

    return {}
