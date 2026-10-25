# Empirical Benchmarking & Scientific Evaluation

This directory contains the automated evaluation harnesses used to test AutoClean against academic baselines, calculate Defect Resolution Rates (DRR), and measure downstream machine learning utility lifts under the CleanML protocol.

---

## Evaluation Suite Index

| Script / Directory | Role & Justification | Key Metrics Evaluated |
| :--- | :--- | :--- |
| **`run_full_scale_complete_benchmarks.py`** | **Primary Benchmark (Flagship)**: Evaluates 100% untruncated rows (zero row-slicing) across the 5 core showcase datasets (Online Retail 541k, Adult Census 32k, Dirty Cafe Sales 10k, Baran Movies 7.3k, Baran Beers 2.4k). | Wall-clock execution time, repaired cell counts, Defect Resolution Rate (DRR), CleanML Downstream ML Delta ($R^2$ / Accuracy / RMSE). |
| **`compare_published.py`** | **Academic Comparative Benchmark**: Benchmarks AutoClean directly against published literature figures from Baran (PVLDB 2020), Raha (SIGMOD 2019), and HoloClean (PVLDB 2017) using verified ground truth. | Precision, Recall, $F_1$ Score, and Harm Rate. |
| **`run_benchmark.py`** | **Ground-Truth Harness**: Executes cell-by-cell ground-truth verification on datasets located in `benchmarks/` (`clean.csv` vs `dirty.csv`). | True Positives, False Positives, False Negatives, Clean Cells Damaged. |
| **`verify_benchmark_numbers.py`** | **Live Metric Verification**: Audits and prints real-time accuracy and cell repair rates without hardcoding or cached values. | Dynamic verification of cell repair counts. |
| **`run_multi_dataset_ml.py`** | **Downstream ML Harness**: Evaluates predictive impact across classification (Telco) and regression benchmarks on held-out test splits. | Accuracy, F1 Score, $R^2$ Variance Explained, RMSE. |
| **`results/`** | **Archival Output Directory**: Stores versioned benchmark JSON and Markdown logs produced by benchmark runs. | Historical empirical run logs. |
