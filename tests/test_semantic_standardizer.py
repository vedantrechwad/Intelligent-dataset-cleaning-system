"""
tests/test_semantic_standardizer.py
Unit tests for AI Semantic Vocabulary Standardizer and Semantic Tie-Breaker.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.intelligence.semantic_standardizer import (
    generate_semantic_vocabulary_proposal,
    semantic_tie_break
)
from cleaner.rules.base import Proposal, CellChange
from cleaner.engine import CleaningEngine


def test_semantic_vocabulary_offline():
    mock_client = MagicMock()
    mock_client.check_availability.return_value = False

    df = pd.DataFrame({"title": ["SWE", "Senior SWE", "Dev", "Lead Dev"] * 5})
    prop = generate_semantic_vocabulary_proposal(df, "title", mock_client)
    assert prop is None


def test_semantic_vocabulary_mapping():
    mock_client = MagicMock()
    mock_client.check_availability.return_value = True
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.model = "qwen2.5:7b"
    mock_client.timeout = 5

    df = pd.DataFrame({
        "role": ["Sr. Dev", "Senior Developer", "Senior Developer", "Jr. Dev", "Junior Developer"] * 10
    })

    fake_response = {
        "response": '{"canonical_mapping": {"Sr. Dev": "Senior Developer", "Jr. Dev": "Junior Developer"}, "reasoning": "Expanded abbreviations."}'
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = fake_response

        prop = generate_semantic_vocabulary_proposal(df, "role", mock_client)
        assert prop is not None
        assert prop.kind == "R9_semantic_vocabulary"
        assert prop.tier == "REVIEW"
        assert prop.columns == ["role"]
        assert len(prop.changes) == 20  # 10 'Sr. Dev' + 10 'Jr. Dev'
        assert any(c.old_value == "Sr. Dev" and c.new_value == "Senior Developer" for c in prop.changes)
        assert any(c.old_value == "Jr. Dev" and c.new_value == "Junior Developer" for c in prop.changes)


def test_semantic_tie_break():
    mock_client = MagicMock()
    mock_client.check_availability.return_value = True
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.model = "qwen2.5:7b"

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": '{"are_same_entity": false, "confidence": 0.99, "explanation": "Pediatrics and Podiatry are distinct specialties."}'
        }

        same, expl = semantic_tie_break("Pediatrics", "Podiatry", "Specialty", mock_client)
        assert same is False
        assert "Pediatrics" in expl or "distinct" in expl


def test_engine_integration_with_semantic_ai():
    mock_client = MagicMock()
    mock_client.check_availability.return_value = True
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.model = "qwen2.5:7b"
    mock_client.timeout = 5

    df = pd.DataFrame({
        "role": ["Sr. Dev", "Senior Developer", "Senior Developer", "Jr. Dev", "Junior Developer"] * 10
    })

    fake_response = {
        "response": '{"canonical_mapping": {"Sr. Dev": "Senior Developer"}, "reasoning": "Normalized."}'
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = fake_response

        engine = CleaningEngine()
        props = engine.generate_proposals(df, enable_semantic_ai=True, ollama_client=mock_client)
        r9_props = [p for p in props if p.kind == "R9_semantic_vocabulary"]
        assert len(r9_props) == 1

        # Test applying proposal
        cleaned_df, diffs = engine.apply(df, r9_props)
        assert "Sr. Dev" not in cleaned_df["role"].values
        assert len(diffs) == 10
