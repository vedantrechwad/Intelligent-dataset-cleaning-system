# Automated Test Suite (Pytest)

This directory contains the automated unit, invariant, and regression tests for **AutoClean**.

---

## Test Suite Execution

Run all 88 unit tests:
```bash
pytest tests/
```

Or run with verbose output:
```bash
pytest -v
```

---

## Module Coverage & Justifications

| Test File | Subsystem Tested | Key Assertions & Invariants |
| :--- | :--- | :--- |
| **`test_phase1.py`** | Rules R1, R2, R3 | Missing sentinel handling, numeric unit stripping, whitespace and casing normalization. |
| **`test_phase2.py`** | Rules R4, R5, R6 | ISO 8601 date parsing, compound column splitting, and approximate functional dependency repairs. |
| **`test_phase3.py`** | Rules R7, R8, Engine Core | Levenshtein variant clustering, duplicate deduplication, and byte-identical invariant preservation. |
| **`test_phase4.py`** | Exporter & Diff Viewer | Valid Python syntax generation for `clean_pipeline.py`, accurate diff cell counts, and rollback integrity. |
| **`test_comprehensive_engine_audit.py`** | Full Engine Pipeline | End-to-end proposal generation, execution contracts, and zero-harm bounds. |
| **`test_human_review_and_intelligence.py`** | Review Workspace & AI | Interactive spreadsheet manual overrides, AI contextual anomaly assistant, natural language rule parser, and AI metric shift explainer. |
| **`test_semantic_standardizer.py`** | Category Harmonizer | $O(1)$ column-level semantic vocabulary clustering via LLM / embedding standardizer. |
| **`test_confidence_quality.py`** | Quality Scoring Engine | Continuous hyperbolic penalty scaling, Defect Resolution Rate (DRR), and zero premature saturation clamping. |
| **`test_detection.py`** | Anomaly Detectors | Vectorized detection for types, dates, inconsistencies, and statistical outliers. |
| **`test_loader_profiler.py`** | Ingestion & Profiler | Invariant-safe CSV/XLSX string loading, profiler speed, and null sentinel identification. |
| **`test_regression_fixes.py`** | Regression Fixes | Prevents recurrence of known edge-case bugs (century date shifts, mixed numeric strings, etc.). |
| **`test_universal_enhancements.py`** | Invariant Bounds | Mathematical verification that untouched cells remain 100% byte-identical. |
| **`test_cleaning_pipeline.py`** | Standalone Script Execution | Validates that exported Python cleaning scripts run and replicate identical results in isolation. |
| **`test_compare_published.py`** | Evaluation Mathematics | Precision, Recall, F1, and Harm Rate calculation formulas. |
| **`test_evaluation_corruption.py`** | Synthetic Injection | Verifies corruption injection manifests and held-out evaluation splits. |
| **`test_overhaul_verification.py`** | Large Table Invariants | Invariant verification on real-world Kaggle datasets (Car Dekho, Customer Churn). |
