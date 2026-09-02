# 🛡️ Intelligent Dataset Quality Assessment & Automated Cleaning System

A production-style, modular, full-stack Python application that automatically profiles tabular datasets (CSV/XLSX), detects data quality anomalies across multiple statistical and domain dimensions, executes calibrated confidence-scored automated cleaning, facilitates human-in-the-loop validation, and rigorously benchmarks performance using a ground-truth data corruption engine and downstream ML evaluation.

---

## 📋 Table of Contents
- [1. Project Overview & Problem Statement](#1-project-overview--problem-statement)
- [2. System Architecture](#2-system-architecture)
- [3. Key Architectural Principles](#3-key-architectural-principles)
- [4. Technology Stack](#4-technology-stack)
- [5. Installation & Setup](#5-installation--setup)
- [6. Local Ollama Integration](#6-local-ollama-integration)
- [7. Supported Data Quality Issues](#7-supported-data-quality-issues)
- [8. Multi-Dimensional Quality Scoring](#8-multi-dimensional-quality-scoring)
- [9. Confidence Engine & Decision Routing](#9-confidence-engine--decision-routing)
- [10. Human-in-the-Loop Review Queue](#10-human-in-the-loop-review-queue)
- [11. Scientific Evaluation & Benchmarking](#11-scientific-evaluation--benchmarking)
- [12. Downstream Machine Learning Evaluation](#12-downstream-machine-learning-evaluation)
- [13. Running Pytest Suite](#13-running-pytest-suite)
- [14. Limitations & Conservatism](#14-limitations--conservatism)

---

## 1. Project Overview & Problem Statement
Data engineering and machine learning workflows are frequently bottlenecked by poor data quality: missing representations, inconsistent text casing, spelling typos in categories, invalid domain values (such as negative ages), impossible dates (such as `2025-02-30` or `22/13/2020`), and severe numerical outliers.

Traditional tools either blindly drop rows or rely solely on naive heuristics. Conversely, blindly sending entire datasets to an LLM is slow, expensive, and risks silent hallucinations.

This application implements a **hybrid architecture**:
- **Deterministic Python Core** (Pandas, SciPy, Scikit-learn, RapidFuzz) performs all statistical profiling, anomaly detection, rule validation, and dataset transformations.
- **Optional Local LLM (Ollama with `qwen2.5:7b`)** is queried strictly for semantic ambiguity and categorical typo confirmation with structured JSON schema validation.
- **Clear Distinction Between Detection and Correction Confidence**: The system may be 100% confident a date is invalid, but 0% confident what the true date was, routing it to a strict **Human-in-the-Loop Review Queue**.

---

## 2. System Architecture

```
                       USER UPLOADS DATASET (CSV / XLSX)
                                      ↓
                         DATASET PROFILING & SCHEMA
                                      ↓
                        DATA QUALITY ISSUE DETECTION
    ┌─────────────────────────────────┼─────────────────────────────────┐
    ↓                                 ↓                                 ↓
Statistical / Rules              Outlier Ensemble               Categorical / Types
(Missing, Dups, Ranges, Dates)  (IQR + Z-score + IsoForest)     (RapidFuzz Typos, Casing)
    └─────────────────────────────────┬─────────────────────────────────┘
                                      ↓
                     OPTIONAL LOCAL OLLAMA REASONING
                    (Structured JSON Contextual Query)
                                      ↓
                               DECISION ENGINE
                                      ↓
                Is Correction Confidence >= Threshold (0.95)?
                               /             \
                             YES              NO
                              ↓                ↓
                      AUTO-CORRECT       FLAG FOR HUMAN REVIEW
                              ↓                ↓
                              └───────┬────────┘
                                      ↓
                           CLEANING PIPELINE ENGINE
                                      ↓
                     POST-CLEANING VALIDATION & RE-PROFILE
                                      ↓
        CLEANED DATASET (CSV/XLSX) + JSON AUDIT LOG + HTML REPORT
```

---

## 3. Key Architectural Principles
1. **Raw Data Immutability**: The uploaded raw dataset is never altered directly; all operations execute on an isolated working copy with a complete audit trail.
2. **Deterministic Data Manipulation**: LLMs never blindly rewrite dataset files. Python algorithms execute all data cleaning operations deterministically.
3. **Graceful Offline Fallback**: If Ollama is offline or unavailable, the application operates with 100% functionality using statistical and frequency heuristics.
4. **Context-Only LLM Queries**: Only structured summaries of ambiguous cells (column, suspicious value, candidate target, top 10 frequent values) are sent to the LLM.

---

## 4. Technology Stack
- **Core / Backend**: Python 3.11+
- **Data Engineering**: Pandas, NumPy
- **Statistics**: SciPy
- **Machine Learning & Anomaly Detection**: Scikit-learn (Isolation Forest, Random Forest Classifier/Regressor)
- **String Similarity & Typo Clustering**: RapidFuzz
- **Excel Ingestion & Export**: OpenPyXL
- **Local LLM Engine**: Ollama (HTTP API on `http://127.0.0.1:11434`)
- **Dashboard UI**: Streamlit
- **Visualizations**: Plotly Express & Plotly Graph Objects
- **Testing**: Pytest
- **Configuration**: PyYAML, python-dotenv

---

## 5. Installation & Setup

### 1. Clone & Navigate to Workspace
```bash
cd "Dataset cleaning system"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 6. Local Ollama Integration

The application automatically checks for an active Ollama instance at `http://127.0.0.1:11434`.

### Recommended Local Models
1. **`qwen2.5:7b` (Recommended & Default)**:
   - Superior JSON schema compliance, multilingual normalization, and fast inference.
   - Install command: `ollama pull qwen2.5:7b`
2. **`llama3.1:8b`**:
   - Strong instruction following and linguistic consistency.
   - Install command: `ollama pull llama3.1:8b`
3. **`qwen2.5:14b`**:
   - Premium reasoning accuracy for complex multi-column constraints (requires >= 12GB VRAM).
   - Install command: `ollama pull qwen2.5:14b`
4. **`mistral:7b`**:
   - Lightweight, swift baseline.
   - Install command: `ollama pull mistral:7b`

---

## 7. Supported Data Quality Issues

| Issue Category | Detection Technique | Correction Policy | Default Routing |
| :--- | :--- | :--- | :--- |
| **Missing Values** | `NaN`, `None`, empty, whitespace, and tokens (`null`, `N/A`, `unknown`) | Configurable Imputation (Median / Mean / Mode / KNN) | Auto / Configurable |
| **Exact Duplicates** | Vectorized tuple hash matching | Remove duplicates keeping first | `AUTO_CORRECT` (1.0) |
| **Near Duplicates** | RapidFuzz token sort ratio (>= 90%) | Flag for review | `HUMAN_REVIEW_REQUIRED` |
| **Invalid Ranges** | Domain boundary rules (e.g. `age < 0` or `> 120`) | Strict Human Input Validation | `HUMAN_REVIEW_REQUIRED` |
| **Invalid Dates** | Unparseable strings, invalid days (Feb 30), month > 12 | Strict Date Format Validation | `HUMAN_REVIEW_REQUIRED` |
| **Type Inconsistencies** | Numeric strings (`"25"` vs `"Thirty"`) | Safe Numeric Cast or Word Map | Auto (`"25"`) / Review (`"Thirty"`) |
| **Whitespace / Casing** | Regex normalization & Title Case alignment | Canonical string replacement | `AUTO_CORRECT` (>=0.98) |
| **Spelling Typos** | RapidFuzz ratio + Frequency ratio + Ollama | Dominant category replacement | Auto (>=0.95) / Review |
| **Numerical Outliers** | Ensemble (IQR 1.5x + Z-score \|z\|>3 + Isolation Forest) | User policy (`flag_only`, `cap`, `remove`) | Flagged by default |

---

## 8. Multi-Dimensional Quality Scoring

The reproducible **Data Quality Score (0–100)** is computed across 5 explainable dimensions:

$$\text{Quality Score} = w_1 \cdot \text{Completeness} + w_2 \cdot \text{Consistency} + w_3 \cdot \text{Validity} + w_4 \cdot \text{Uniqueness} + w_5 \cdot \text{Anomaly Quality}$$

- **Completeness**: $100 \times (1 - \frac{\text{missing cells}}{\text{total cells}})$
- **Uniqueness**: $100 \times (1 - \frac{\text{duplicate rows}}{\text{total rows}})$
- **Validity**: $100 \times (1 - \frac{\text{invalid dates/ranges}}{\text{total cells}})$
- **Consistency**: $100 \times (1 - \frac{\text{typos/casing/whitespace}}{\text{total cells}})$
- **Anomaly Quality**: $100 \times (1 - \frac{\text{extreme outliers}}{\text{total numeric cells}})$

---

## 9. Confidence Engine & Decision Routing

Confidence is calculated independently for **detection** versus **correction**.

### Categorical Typo Correction Confidence Formula
$$\text{Confidence} = 0.50 \times \left(\frac{\text{Similarity}}{100}\right) + 0.30 \times \left(\frac{\text{Freq}_{\text{dominant}} - \text{Freq}_{\text{candidate}}}{\text{Freq}_{\text{dominant}} + \text{Freq}_{\text{candidate}}}\right) + 0.20 \times \text{Contextual Evidence}$$

### Decision Thresholds
- **$\ge 0.95$**: `AUTO_CORRECT` (Safe to apply automatically)
- **$0.70 \le \text{Confidence} < 0.95$**: `SUGGEST_REVIEW` (Candidate proposed in queue)
- **$< 0.70$ or Domain Constraint**: `HUMAN_REVIEW_REQUIRED` (User input strictly required)

---

## 10. Human-in-the-Loop Review Queue
For values where the system knows a value is wrong but cannot guess the true value (e.g. `Age = -5` or `Join Date = 22/13/2020`):
1. The issue is displayed in the **Review Queue** with detected reason and candidate value.
2. The user can **Accept**, **Reject**, or provide a **Custom Overriding Value**.
3. All inputs are strictly validated before committing (e.g., entered dates must parse as genuine calendar dates; numbers must satisfy configured domain ranges).

---

## 11. Scientific Evaluation & Benchmarking
The project includes a **Data Corruption Engine** to inject ground-truth anomalies into clean benchmark datasets:
- Tracks exact coordinates `(row, col)` of every injected error.
- Measures **Precision, Recall, and F1 Score** against true positives, false positives, and false negatives.
- Computes **Restoration Accuracy**:
  $$\text{Accuracy} = \frac{\text{Number Correctly Restored}}{\text{Automatic Corrections Attempted}}$$
- Explicitly tracks human-assisted corrections and unresolved errors separately from automated accuracy.

---

## 12. Downstream Machine Learning Evaluation
Trains identical Random Forest Classifiers/Regressors on the **Corrupted vs. Cleaned** datasets under locked random seeds, identical splits, and robust preprocessing to measure:
- Classification: Accuracy, Weighted F1, Precision, Recall.
- Regression: $R^2$ Score, MAE, RMSE.
- *Note:* The report explicitly states that cleaning improves training stability, but does not guarantee test accuracy improvements on all distributions.

---

## 13. Running Pytest Suite
Run the test suite:
```bash
python -m pytest tests/ -v
```

---

## 14. Limitations & Conservatism
- The system is intentionally conservative: it will never invent a date or impute values when confidence is low.
- Domain rules (such as realistic age ranges) should be configured in `config.yaml` to match specific business domains.
- Recommended dataset size for interactive Streamlit session: up to 50,000 rows.
