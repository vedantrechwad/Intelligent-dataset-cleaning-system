import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.utils.helpers import NpEncoder, logger

class AuditLogger:
    """
    Tracks and exports immutable, fine-grained audit trails
    of every quality anomaly, automated intervention, and human-in-the-loop decision.
    """
    def __init__(self, dataset_name: str = "dataset"):
        self.dataset_name = dataset_name
        self.created_at = datetime.now().isoformat()
        self.detected_issues: List[Dict[str, Any]] = []
        self.applied_actions: List[Dict[str, Any]] = []

    def log_detected_issues(self, issues: Optional[List[Dict[str, Any]]]) -> None:
        self.detected_issues = issues if issues is not None else []

    def log_applied_actions(self, actions: Optional[List[Dict[str, Any]]]) -> None:
        self.applied_actions = actions if actions is not None else []

    def generate_audit_report(self) -> Dict[str, Any]:
        """Compiles the complete audit report document."""
        detected = self.detected_issues if self.detected_issues is not None else []
        applied = self.applied_actions if self.applied_actions is not None else []
        return {
            "system": "Intelligent Dataset Quality Assessment & Cleaning System",
            "dataset_name": self.dataset_name,
            "timestamp": self.created_at,
            "total_detected_issues": len(detected),
            "total_applied_actions": len(applied),
            "summary_by_action": self._summarize_actions(),
            "applied_actions": applied,
            "detected_issues": detected
        }

    def _summarize_actions(self) -> Dict[str, int]:
        counts = {}
        for act in (self.applied_actions or []):
            if isinstance(act, dict):
                action_name = act.get("action", "unknown")
            else:
                action_name = "unknown"
            counts[action_name] = counts.get(action_name, 0) + 1
        return counts

    def to_json(self, indent: int = 2) -> str:
        """Serializes the audit report to a valid JSON string."""
        return json.dumps(self.generate_audit_report(), cls=NpEncoder, indent=indent)
