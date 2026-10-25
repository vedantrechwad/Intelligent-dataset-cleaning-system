# Developer & Operational Scripts

This directory contains utility scripts for repository maintenance, test fixture generation, and dataset ingestion.

---

## Script Directory & Justifications

| Script | Purpose & Justification | Command |
| :--- | :--- | :--- |
| **`verify_system.py`** | **Developer Sanity Check**: Runs a fast (<2s) end-to-end verification of table ingestion, invariant profiling, rule generation, cell diffs, and standalone script export. Run before pushing to git. | `python scripts/verify_system.py` |
| **`generate_sample_data.py`** | **Fixture Generator**: Creates the synthetic telecom churn mock datasets in `sample_data/` required by the automated test suite. | `python scripts/generate_sample_data.py` |
| **`download_benchmarks.py`** | **Academic Fetcher**: Idempotently downloads Raha/Baran benchmark pairs (`dirty.csv` and `clean.csv`) into `benchmarks/`. | `python scripts/download_benchmarks.py` |
| **`download_10_more_dirty_datasets.py`** | **Kaggle Fetcher**: Downloads curated dirty benchmark CSVs into `data/dirty_benchmarks/`. | `python scripts/download_10_more_dirty_datasets.py` |
| **`download_5_large_scale_datasets.py`** | **Large-Scale Fetcher**: Downloads high-volume datasets (>30k to >500k rows) into `data/dirty_large_scale/`. | `python scripts/download_5_large_scale_datasets.py` |
