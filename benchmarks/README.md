# Benchmark Datasets

This directory contains research benchmark datasets from the Raha/Baran benchmark suite.

## Data Source
- Repository: [BigDaMa/raha](https://github.com/BigDaMa/raha)
- Datasets Path: `https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/<dataset>/<file>`
- Reference Papers:
  - *Baran: Effective Error Correction via a Unified Context Representation and Transfer Learning* (PVLDB 2020)
  - *Raha: A Configuration-Free Error Detection System* (SIGMOD 2019)
  - *HoloClean: Holistic Data Repairs with Probabilistic Inference* (PVLDB 2017)

## Verified Dataset Shapes
| Dataset | dirty.csv Shape | clean.csv Shape | Role in Evaluation | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **hospital** | (1000, 20) | (1000, 20) | Development | Includes 1 index column (excluded from cell counts) |
| **beers** | (2410, 11) | (2410, 11) | Development | Includes 1 index column (excluded from cell counts) |
| **tax** | (200000, 15) | (200000, 15) | Held-Out | Semi-synthetic IRS tax benchmark (Baran PVLDB 2020) |
| **flights** | (2376, 7) | (2376, 7) | Held-Out (Optional) | Real flight delays/cancellations (external radar data) |

## Downloading & Verification
To download or verify datasets idempotently:
```bash
python scripts/download_benchmarks.py
```
