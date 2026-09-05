from typing import List, Dict, Any, Optional
import pandas as pd
from src.intelligence.ollama_client import OllamaClient
from src.intelligence.confidence_engine import calculate_category_correction_confidence, route_decision
from src.utils.helpers import logger

class ReasoningEngine:
    """
    Intelligent reasoning orchestrator.
    Combines rule-based / statistical heuristics with optional Ollama reasoning.
    Enhances ambiguous issues with LRU caching for ultra-fast performance.
    """
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama = ollama_client or OllamaClient()
        self._cache = {}

    def enrich_issues(
        self,
        issues: List[Dict[str, Any]],
        df: pd.DataFrame,
        use_ollama: bool = True,
        max_llm_queries: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Enriches detected issues with contextual reasoning and updated confidence.
        Queries Ollama only for ambiguous or borderline issues, utilizing query caching.
        """
        is_ollama_ready = use_ollama and self.ollama.check_availability()
        if is_ollama_ready:
            logger.info("Ollama is available. Contextual LLM reasoning active.")
        else:
            logger.info("Operating in deterministic heuristic mode (Ollama offline or disabled).")

        enriched = []
        queries_made = 0

        for issue in issues:
            itype = issue.get("issue_type")
            
            # Deterministic issues require zero LLM calls
            if itype in ["whitespace_inconsistency", "exact_duplicate", "invalid_range", "invalid_date", "missing_value"]:
                enriched.append(issue)
                continue

            # Candidate for LLM reasoning: borderline spelling typos or ambiguous conversions
            if is_ollama_ready and queries_made < max_llm_queries and itype in ["spelling_typo", "type_inconsistency", "near_duplicate"]:
                col = issue.get("column")
                raw_val = issue.get("original_value")
                suggested_val = issue.get("suggested_value")

                cache_key = (col, str(raw_val), str(suggested_val))
                if cache_key in self._cache:
                    llm_result = self._cache[cache_key]
                else:
                    if col in df.columns:
                        top_vals = [str(x) for x in df[col].dropna().value_counts().head(5).index]
                        llm_result = self.ollama.reason_about_issue(
                            column_name=col,
                            column_type=str(df[col].dtype),
                            detected_value=raw_val,
                            frequent_values=top_vals,
                            candidate_correction=str(suggested_val) if suggested_val else None
                        )
                        self._cache[cache_key] = llm_result
                        queries_made += 1
                    else:
                        llm_result = None

                if llm_result:
                    issue["detection_method"].append("ollama_reasoning")
                    issue["llm_reasoning"] = llm_result.get("reason")
                    
                    if itype == "spelling_typo":
                        current_conf = issue.get("correction_confidence", 0.8)
                        blended = (0.6 * current_conf) + (0.4 * llm_result.get("confidence", 0.8))
                        issue["correction_confidence"] = round(blended, 3)
                        
                        if llm_result.get("corrected_value"):
                            issue["suggested_value"] = llm_result.get("corrected_value")
                            
                        issue["routing_decision"] = route_decision(
                            issue["correction_confidence"], 
                            itype, 
                            issue.get("is_human_review_required", False)
                        )

            enriched.append(issue)

        return enriched

    def generate_issue_explanation(self, issue_type: str, column: str) -> str:
        """
        Generates a very simple, beginner-friendly explanation of an issue type
        and how correcting it improves the dataset. Uses caching to avoid redundant queries.
        """
        if not self.ollama.check_availability():
            return "Explanation unavailable (Ollama offline)."
            
        cache_key = f"explain_{issue_type}_{column}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        prompt = f"""You are a helpful data assistant. Keep your answer under 2 sentences.
We found a data quality issue of type '{issue_type}' in the column '{column}'.
Explain very simply what this issue means, and how fixing it will improve the dataset for future analysis.
"""
        payload = {
            "model": self.ollama.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        
        try:
            import requests
            r = requests.post(f"{self.ollama.host}/api/generate", json=payload, timeout=self.ollama.timeout)
            if r.status_code == 200:
                explanation = r.json().get("response", "").strip()
                self._cache[cache_key] = explanation
                return explanation
        except Exception as e:
            logger.warning(f"Explanation query failed: {e}")
            
        return "Explanation unavailable."
