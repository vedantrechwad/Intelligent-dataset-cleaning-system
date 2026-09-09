# Intelligent Dataset Cleaning System

An automated, AI-augmented data preparation pipeline designed for tabular datasets. This system integrates traditional mathematical anomaly detection with a **Local Large Language Model (LLM)** to provide semantic imputation, reasoning, and explainability, all while guaranteeing 100% data privacy.

---

## 1. System Overview

Data scientists spend up to 80% of their time cleaning and preparing data. Standard open-source cleaners (like `AutoClean`) operate blindly, forcing statistical averages on missing data and ignoring context. Cloud-based enterprise solutions (like Zoho DataPrep) require uploading sensitive data to external servers.

This project bridges the gap by providing a **local, Human-in-the-Loop (HITL)** architecture. It uses strict mathematical formulas (Pandas/SciPy) for deterministic errors and delegates ambiguous, semantic errors to a local LLM (`qwen2.5:7b` via Ollama).

### Key Differentiators
1. **Zero Data Leakage**: The entire pipeline, including the LLM reasoning engine, runs locally.
2. **Context-Aware Imputation**: The LLM infers missing values by reading the context of the *entire row* rather than defaulting to statistical modes.
3. **Conversational RAG**: Users can chat with their cleaned dataset using a local Retrieval-Augmented Generation pipeline.
4. **Reproducible Auditing**: Every modified cell is logged in a centralized JSON audit trail, ensuring scientific reproducibility.

---

## 2. Core Architecture & Workflow

The cleaning pipeline executes across five distinct stages:

### A. Mathematical Profiling
When a dataset is uploaded, the system generates a statistical profile:
- Analyzes cardinality, data types, standard deviation, and missingness.
- Calculates a **Geometric Data Quality Score (0-100%)** that heavily penalizes structural inconsistencies.
  
  $$\text{Quality Score} = w_1(\text{Completeness}) \times w_2(\text{Consistency}) \times w_3(\text{Validity}) \times w_4(\text{Uniqueness}) \times w_5(\text{Anomalies})$$

### B. Deterministic Anomaly Detection
The detection engine mathematically sweeps the dataset for standard errors:
- **Outliers**: Detected using Interquartile Range (IQR) bounds ($1.5 \times IQR$) and Z-Scores ($|Z| > 3$).
- **Near-Duplicates**: Uses `RapidFuzz` to compute the **Levenshtein Distance** between row vectors to find clusters of near-duplicate entries (e.g., $\ge 90\%$ similarity).
- **Inconsistencies**: Flags structural typos, case-sensitivity issues, and whitespace padding.

### C. The Local LLM Engine (Ollama)
For ambiguous issues, the system queries a local instance of Ollama:
- **Dynamic Domain Inference**: Infers logical constraints (e.g., "Tenure cannot be negative") dynamically based on the dataset schema.
- **AI Issue Explainability**: Generates plain-English explanations for anomalies and why they should be fixed.
- **Context-Aware Imputation**: If a value is missing, the row's context is sent to the LLM to semantically infer the missing value (e.g., inferring a missing "Internet Service" type based on the customer's "Monthly Charges").
- **Dataset Q&A Chat**: A chat interface that receives the statistical profile and mathematical anomaly list, allowing natural language queries against the dataset.

### D. Human-in-the-Loop Validation
The system calculates a **Correction Confidence Score** ($0.0$ to $1.0$) for every anomaly:
- **$\ge 0.95$**: `AUTO_CORRECT` (e.g., Safe whitespace trimming).
- **$0.70 \le \text{Confidence} < 0.95$**: `SUGGEST_REVIEW` (e.g., LLM Imputation or minor spelling fixes).
- **$< 0.70$**: `HUMAN_REVIEW_REQUIRED` (e.g., Heavy numerical outliers).

Users are provided an interactive Streamlit Data Editor to accept, bulk-apply, or individually modify suggested corrections before execution.

---

## 3. Technology Stack

- **Core & Data Engineering**: Python 3.11+, Pandas, NumPy
- **Statistics & Math**: SciPy, Scikit-Learn (Isolation Forest, KNN Imputer)
- **Fuzzy Matching**: RapidFuzz (Levenshtein Token Sort Ratio)
- **Local AI / LLM**: Ollama (`qwen2.5:7b` recommended)
- **Frontend / UI**: Streamlit
- **Testing**: Pytest

---

## 4. Installation & Setup

### 1. Requirements
- Python 3.11 or higher
- [Ollama](https://ollama.ai/) installed locally on your machine.

### 2. Download Recommended LLM
Open your terminal and run:
```bash
ollama pull qwen2.5:7b
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Scientific Benchmarking

The project includes an automated evaluation suite (`test_regression_fixes.py`) designed to measure restoration accuracy on corrupted datasets.

1. **Data Corruption Engine**: Injects ground-truth anomalies (typos, exact duplicates, nulls) into a clean benchmark dataset.
2. **Restoration Evaluation**: Measures Precision, Recall, and F1 Score by verifying if the cleaner successfully restored the exact original cell value.

---

## 6. Project Modules

- `app.py`: Main Streamlit UI and Tab orchestration.
- `src/profiling/`: Generates the metadata summary and quality metrics.
- `src/detection/`: Contains mathematical detectors (`outlier_detector.py`, `duplicate_detector.py`).
- `src/intelligence/`: Interfaces with Ollama for Semantic Imputation (`llm_imputer.py`) and RAG Chat (`chat_engine.py`).
- `src/cleaning/`: Executes deterministic fixes and manages the centralized `AuditLogger`.
- `tests/`: Pytest suite for regression testing.
