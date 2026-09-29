# Benchmark Results (Legacy Pipeline (Baseline-v0))
**Timestamp**: 2026-09-30T00:00:11.787215

| Dataset | Setting | Issues / Proposals | Errors (|E|) | Fixed | Missed | Wrong | Damaged | Recall | Precision | Harm Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| beers | AUTO | 14915 | 4362 | 17 | 1512 | 2833 | 876 | 0.39% | 0.46% | 3.96% |
| beers | AUTO+REVIEW | 14915 | 4362 | 17 | 1281 | 3064 | 1978 | 0.39% | 0.34% | 8.93% |
| hospital | AUTO | 19053 | 509 | 202 | 277 | 30 | 2008 | 39.69% | 9.02% | 10.30% |
| hospital | AUTO+REVIEW | 19053 | 509 | 241 | 174 | 94 | 3296 | 47.35% | 6.64% | 16.91% |
| flights | AUTO | 13005 | 4920 | 45 | 2570 | 2305 | 318 | 0.91% | 1.69% | 2.72% |
| flights | AUTO+REVIEW | 13005 | 4920 | 45 | 2570 | 2305 | 324 | 0.91% | 1.68% | 2.77% |
| rayyan | AUTO | 7541 | 948 | 0 | 869 | 79 | 1757 | 0.00% | 0.00% | 17.48% |
| rayyan | AUTO+REVIEW | 7541 | 948 | 0 | 869 | 79 | 1834 | 0.00% | 0.00% | 18.25% |
| telco | AUTO | 63592 | 449 | 414 | 0 | 35 | 11 | 92.20% | 90.00% | 0.01% |
| telco (strict) | AUTO | 63592 | 449 | 223 | 0 | 226 | 41386 | 49.67% | 0.53% | 28.07% |
| telco | AUTO+REVIEW | 63592 | 449 | 414 | 0 | 35 | 11 | 92.20% | 90.00% | 0.01% |
| telco (strict) | AUTO+REVIEW | 63592 | 449 | 223 | 0 | 226 | 41386 | 49.67% | 0.53% | 28.07% |

### Changes Applied Per Rule / Issue Type
- **beers (AUTO)**: `{'spelling_typo': 1841, 'whitespace_inconsistency': 1, 'casing_inconsistency': 898, 'missing_value': 1199, 'outlier': 3}`
- **beers (AUTO+REVIEW)**: `{'spelling_typo': 3172, 'outlier': 6, 'whitespace_inconsistency': 1, 'casing_inconsistency': 897, 'missing_value': 1199}`
- **hospital (AUTO)**: `{'spelling_typo': 1074, 'casing_inconsistency': 790, 'whitespace_inconsistency': 78, 'type_inconsistency': 986}`
- **hospital (AUTO+REVIEW)**: `{'spelling_typo': 2268, 'casing_inconsistency': 995, 'whitespace_inconsistency': 78, 'type_inconsistency': 1000}`
- **flights (AUTO)**: `{'spelling_typo': 318, 'casing_inconsistency': 54, 'whitespace_inconsistency': 35, 'missing_value': 2312}`
- **flights (AUTO+REVIEW)**: `{'spelling_typo': 324, 'casing_inconsistency': 59, 'whitespace_inconsistency': 35, 'missing_value': 2312}`
- **rayyan (AUTO)**: `{'spelling_typo': 13, 'whitespace_inconsistency': 103, 'casing_inconsistency': 43, 'type_inconsistency': 2, 'missing_value': 1684, 'outlier': 76}`
- **rayyan (AUTO+REVIEW)**: `{'spelling_typo': 14, 'outlier': 152, 'whitespace_inconsistency': 103, 'casing_inconsistency': 44, 'type_inconsistency': 2, 'missing_value': 1684}`
- **telco (AUTO)**: `{'spelling_typo': 83, 'whitespace_inconsistency': 171, 'casing_inconsistency': 148, 'type_inconsistency': 41769, 'exact_duplicate': 30, 'missing_value': 71}`
- **telco (strict) (AUTO)**: `{'spelling_typo': 83, 'whitespace_inconsistency': 171, 'casing_inconsistency': 148, 'type_inconsistency': 41769, 'exact_duplicate': 30, 'missing_value': 71}`
- **telco (AUTO+REVIEW)**: `{'spelling_typo': 83, 'whitespace_inconsistency': 171, 'casing_inconsistency': 148, 'type_inconsistency': 41769, 'exact_duplicate': 30, 'missing_value': 71}`
- **telco (strict) (AUTO+REVIEW)**: `{'spelling_typo': 83, 'whitespace_inconsistency': 171, 'casing_inconsistency': 148, 'type_inconsistency': 41769, 'exact_duplicate': 30, 'missing_value': 71}`