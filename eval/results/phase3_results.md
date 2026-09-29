# Benchmark Results (New Rule Engine)
**Timestamp**: 2026-09-30T00:52:40.313288

| Dataset | Setting | Issues / Proposals | Errors (|E|) | Fixed | Missed | Wrong | Damaged | Recall | Precision | Harm Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| beers | AUTO | 9 | 4362 | 2270 | 2092 | 0 | 0 | 52.04% | 100.00% | 0.00% |
| beers | AUTO+REVIEW | 9 | 4362 | 4357 | 5 | 0 | 52 | 99.89% | 98.82% | 0.23% |
| hospital | AUTO | 39 | 509 | 0 | 509 | 0 | 0 | 0.00% | 0.00% | 0.00% |
| hospital | AUTO+REVIEW | 39 | 509 | 407 | 102 | 0 | 20 | 79.96% | 95.32% | 0.10% |
| flights | AUTO | 0 | 4920 | 0 | 4920 | 0 | 0 | 0.00% | 0.00% | 0.00% |
| flights | AUTO+REVIEW | 0 | 4920 | 0 | 4920 | 0 | 0 | 0.00% | 0.00% | 0.00% |
| rayyan | AUTO | 6 | 948 | 0 | 948 | 0 | 5 | 0.00% | 0.00% | 0.05% |
| rayyan | AUTO+REVIEW | 6 | 948 | 0 | 948 | 0 | 5 | 0.00% | 0.00% | 0.05% |
| telco | AUTO | 33 | 449 | 183 | 206 | 60 | 11 | 40.76% | 72.05% | 0.01% |
| telco | AUTO+REVIEW | 33 | 449 | 198 | 191 | 60 | 11 | 44.10% | 73.61% | 0.01% |

### Changes Applied Per Rule / Issue Type
- **beers (AUTO)**: `{'R2_numeric': 2270}`
- **beers (AUTO+REVIEW)**: `{'R1_missing_tokens': 1005, 'R2_numeric': 3103, 'R5_compound_split': 246, 'R6_dependency_repair': 55}`
- **hospital (AUTO+REVIEW)**: `{'R6_dependency_repair': 416, 'R7_category_variants': 11}`
- **rayyan (AUTO)**: `{'R1_missing_tokens': 2, 'R3_whitespace_case': 3}`
- **rayyan (AUTO+REVIEW)**: `{'R1_missing_tokens': 2, 'R3_whitespace_case': 3}`
- **telco (AUTO)**: `{'R1_missing_tokens': 71, 'R3_whitespace_case': 183}`
- **telco (AUTO+REVIEW)**: `{'R1_missing_tokens': 71, 'R3_whitespace_case': 183, 'R7_category_variants': 15, 'R8_exact_duplicates': 30}`