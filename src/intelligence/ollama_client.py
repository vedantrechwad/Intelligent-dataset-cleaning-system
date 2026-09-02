import os
import json
import time
import subprocess
from typing import Dict, Any, Optional, List
import requests
from src.utils.helpers import load_config, logger

RECOMMENDED_MODELS = [
    {
        "name": "qwen2.5:7b",
        "description": "Recommended. Outstanding JSON schema adherence, structured reasoning, and high performance.",
        "parameters": "7.6B",
        "vram_required": "~5 GB"
    },
    {
        "name": "llama3.1:8b",
        "description": "Top-tier general purpose instruction following and linguistic consistency.",
        "parameters": "8.0B",
        "vram_required": "~5.5 GB"
    },
    {
        "name": "qwen2.5:14b",
        "description": "Premium tier. Unmatched reasoning fidelity for complex multi-column relationships.",
        "parameters": "14.7B",
        "vram_required": "~10-12 GB"
    },
    {
        "name": "mistral:7b",
        "description": "Lightweight, swift reasoning baseline.",
        "parameters": "7.2B",
        "vram_required": "~4.5 GB"
    }
]

class OllamaClient:
    """
    Robust, fault-tolerant client for local Ollama instances.
    Provides automatic health checks, optional background auto-launch,
    and schema-validated JSON generation for data quality reasoning.
    """
    def __init__(self, host: str = None, model: str = None, timeout: int = 15):
        cfg = load_config().get("ollama", {})
        self.host = host or os.environ.get("OLLAMA_HOST") or cfg.get("host", "http://127.0.0.1:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL") or cfg.get("model", "qwen2.5:7b")
        self.timeout = timeout or cfg.get("timeout_seconds", 15)
        self._is_available = None

    def auto_start_if_needed(self) -> bool:
        """Attempts to ping Ollama, and if not running, attempts to launch 'ollama serve'."""
        if self.check_availability():
            return True

        logger.info("Ollama is not running. Attempting to start 'ollama serve' in background...")
        try:
            # On Windows, start detached process
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
            )
            # Wait briefly for startup
            for _ in range(5):
                time.sleep(1)
                if self.check_availability():
                    logger.info("Ollama successfully auto-started!")
                    return True
        except Exception as e:
            logger.warning(f"Could not auto-start Ollama: {e}")

        return False

    def check_availability(self) -> bool:
        """Pings Ollama server on 127.0.0.1 to verify active connection."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            self._is_available = (r.status_code == 200)
            return self._is_available
        except Exception:
            self._is_available = False
            return False

    def get_installed_models(self) -> List[str]:
        """Lists all downloaded models on the local Ollama instance."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=4)
            if r.status_code == 200:
                data = r.json()
                return [m.get("name") for m in data.get("models", [])]
        except Exception as e:
            logger.debug(f"Failed to fetch models: {e}")
        return []

    def reason_about_issue(
        self,
        column_name: str,
        column_type: str,
        detected_value: Any,
        frequent_values: List[str],
        candidate_correction: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Sends structured context (NEVER the whole dataset) to Ollama
        to evaluate ambiguity, confirm typos, and recommend actions.
        Enforces valid JSON response with fallbacks.
        """
        if not self.check_availability():
            return None

        prompt = f"""You are an expert data quality validation assistant.
Evaluate the following suspected data anomaly in a tabular dataset:

Column Name: "{column_name}"
Column Type: "{column_type}"
Detected Suspicious Value: "{detected_value}"
Frequent Valid Values in Column: {json.dumps(frequent_values[:10])}
Suggested Candidate Correction: "{candidate_correction or 'None'}"

Determine:
1. Is this value genuinely a data entry error or inconsistency?
2. Is automatic correction safe, or does it require human review?
3. What is the recommended corrected value?
4. What is your confidence score between 0.00 and 1.00?
5. Provide a brief 1-sentence technical explanation.

Respond ONLY with a valid JSON object strictly matching this schema:
{{
    "is_likely_error": true,
    "recommended_action": "replace",
    "corrected_value": "CanonicalValue",
    "confidence": 0.96,
    "reason": "Clear typo of dominant category."
}}
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        try:
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            if r.status_code == 200:
                raw_response = r.json().get("response", "")
                parsed = json.loads(raw_response)
                
                # Validate schema
                if "is_likely_error" in parsed and "confidence" in parsed:
                    return {
                        "is_likely_error": bool(parsed.get("is_likely_error")),
                        "recommended_action": str(parsed.get("recommended_action", "review")),
                        "corrected_value": parsed.get("corrected_value"),
                        "confidence": float(max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))),
                        "reason": str(parsed.get("reason", "LLM reasoning analysis"))
                    }
        except Exception as e:
            logger.warning(f"Ollama reasoning query failed: {e}")

        return None
