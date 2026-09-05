import os
import yaml
import json
import logging
from typing import Any, Dict
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DatasetCleaner")

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Loads configuration from yaml file, with fallback defaults."""
    if not config_path:
        config_path = DEFAULT_CONFIG_PATH
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Error loading {config_path}: {e}. Using fallback defaults.")
            
    return {
        "ollama": {
            "host": "http://127.0.0.1:11434",
            "model": "qwen2.5:7b",
            "timeout_seconds": 15,
            "auto_start": True
        },
        "thresholds": {
            "auto_correct_confidence": 0.95,
            "suggest_review_confidence": 0.70,
            "category_similarity_threshold": 82.0,
            "near_duplicate_similarity_threshold": 90.0,
            "outlier_zscore_threshold": 3.0,
            "outlier_iqr_multiplier": 1.5,
            "isolation_forest_contamination": 0.05
        },
        "missing_representations": [
            "", " ", "nan", "NaN", "none", "None", "null", "NULL",
            "n/a", "N/A", "na", "NA", "?", "-"
        ],
        "quality_score_weights": {
            "completeness": 0.25,
            "consistency": 0.20,
            "validity": 0.25,
            "uniqueness": 0.15,
            "anomaly_quality": 0.15
        },
        "domain_rules": {
            "age": {"min": 0, "max": 120},
            "percentage": {"min": 0.0, "max": 100.0},
            "salary": {"min": 0.0}
        },
        "cleaning_defaults": {
            "missing_numeric_strategy": "median",
            "missing_categorical_strategy": "mode",
            "outlier_strategy": "flag_only",
            "remove_exact_duplicates": True,
            "normalize_text_case": True,
            "normalize_whitespace": True
        }
    }


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to properly handle NumPy and Pandas data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Period)):
            return str(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)
