import json
from typing import Dict, Any, Optional, List
import requests
from src.utils.helpers import logger
from src.intelligence.ollama_client import OllamaClient
import pandas as pd

def chat_with_dataset(
    client: OllamaClient,
    query: str,
    df: pd.DataFrame,
    profile_stats: Dict[str, Any],
    issues: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Passes a user query and a summary of the dataset to Ollama to generate an answer.
    """
    if not client.check_availability():
        return "⚠️ Local LLM is not available. Please ensure Ollama is running."

    # Prepare dataset summary
    col_info = []
    for col, stats in profile_stats.get("columns", {}).items():
        ctype = stats.get("type", "unknown")
        if ctype == "numeric":
            col_info.append(f"- {col} (Numeric): min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean')}, std={stats.get('std')}")
        else:
            col_info.append(f"- {col} (Categorical): {stats.get('unique_count')} unique values")

    # Aggregate detected issues
    issue_summary = ""
    if issues:
        issue_counts = {}
        for iss in issues:
            itype = iss.get("issue_type", "unknown")
            issue_counts[itype] = issue_counts.get(itype, 0) + 1
        
        issue_summary = "Detected Dataset Anomalies:\n"
        for itype, count in issue_counts.items():
            issue_summary += f"- {itype}: {count} affected rows\n"
        
        # Pull out a few outlier examples if requested
        outliers = [i for i in issues if i.get("issue_type") == "outlier"]
        if outliers:
            issue_summary += f"\nSpecifically, {len(outliers)} outliers were mathematically detected (e.g., values like {outliers[0].get('original_value')} in '{outliers[0].get('column')}').\n"
    else:
        issue_summary = "No anomalies detected or issues not provided."

    # Limit sample rows to avoid token overflow
    sample_df = df.head(3).to_dict(orient="records")

    prompt = f"""You are a professional Data Science Assistant analyzing a tabular dataset.
    
Dataset Summary:
{chr(10).join(col_info)}

{issue_summary}

Sample Rows (First 3):
{json.dumps(sample_df, indent=2)}

User Question:
"{query}"

Instructions:
1. Provide a direct, professional, and helpful answer to the User Question.
2. If the user asks for insights, trends, or summaries, use the provided Dataset Summary to guide your answer.
3. If the question requires executing code (e.g., "plot this"), politely explain that you can only provide textual insights based on the summary.
4. Keep the answer concise and well-formatted using Markdown.
"""

    payload = {
        "model": client.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9
        }
    }

    try:
        r = requests.post(f"{client.host}/api/generate", json=payload, timeout=client.timeout + 15)
        if r.status_code == 200:
            return r.json().get("response", "I could not generate an answer at this time.")
        else:
            return f"⚠️ LLM Error: {r.status_code} - {r.text}"
    except Exception as e:
        logger.warning(f"Ollama chat query failed: {e}")
        return f"⚠️ Connection to LLM failed: {e}"
