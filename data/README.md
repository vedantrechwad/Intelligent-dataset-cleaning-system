# AutoClean Dataset Repository & Catalog

This directory contains the organized dataset corpus used for developing, stress-testing, and benchmarking the **AutoClean** neuro-symbolic data cleaning system.

---

## Directory Architecture

```
data/
├── dirty_benchmarks/        # 22 Curated dirty datasets from academic and Kaggle benchmarks
├── dirty_large_scale/       # 5 High-volume datasets (32,000 to 564,000+ rows)
├── real_world/              # 11 Real-world production & Kaggle datasets across domains
└── downloaded/              # 10 Classical ML reference datasets (UCI / Kaggle)
```

> **Note**: Backward compatibility junctions (`data/dirty_downloads`, `data/dirty_downloads_round2`, and `data/large_scale`) map transparently to their consolidated canonical directories.

---

## 1. Large-Scale Datasets (`data/dirty_large_scale/`)

These datasets test AutoClean's vectorized streaming diagnostics and O(N) cleaning throughput under 100% untruncated workloads.

| File | Rows x Cols | Domain | Key Defect Profile |
| :--- | :--- | :--- | :--- |
| `online_retail_540k.csv` | 541,909 x 8 | E-commerce / Retail | Negative prices, cancelled invoice prefixes (`C536...`), missing `CustomerID`s (135k), leading whitespace |
| `uber_trips_nyc_560k.csv` | 564,574 x 4 | Urban Mobility | Extreme latitude/longitude outliers outside NYC bounding box, malformed ISO timestamp strings |
| `pnw_flights_160k.csv` | 162,049 x 16 | Aviation Logistics | Missing arrival/departure delays, disguised sentinel codes, cancelled flight NaN patterns |
| `sf_payroll_74k.csv` | 74,400 x 9 | Public Compensation | Inconsistent capitalization, trailing whitespace in job titles, corrupted currency strings (`$`, `,`) |
| `chicago_salaries_32k.csv` | 32,062 x 6 | Municipal HR | Hourly vs. annual wage unit collisions, Department casing variants, disguised missing values |

---

## 2. Dirty Benchmark Datasets (`data/dirty_benchmarks/`)

Academic benchmark datasets from the Raha/Baran research suite (SIGMOD 2019 / PVLDB 2020) and Kaggle data cleaning competitions.

| File | Rows x Cols | Source / Origin | Key Defect Profile |
| :--- | :--- | :--- | :--- |
| `dirty_cafe_sales.csv` | 10,000 x 8 | Kaggle (shaadyalii) | Corrupted string sentinels (`UNKNOWN`, `ERROR`) injected into numeric total/quantity columns |
| `movies_dirty.csv` | 7,390 x 17 | Baran Benchmark | Inconsistent runtime formats (`95 min`, `1h 35m`), currency gross string noise, malformed release dates |
| `beers_dirty.csv` | 2,410 x 11 | Baran Benchmark | Corrupted text tokens, style inconsistencies, missing ABV percentages |
| `dirty_tax.csv` | 200,000 x 15 | Baran Benchmark (IRS) | Zip code formatting errors, state abbreviation corruptions, married filing dependency rule violations |
| `dirty_hospital.csv` | 1,000 x 20 | Baran / HoloClean | Provider numbers, hospital names with casing errors, inconsistent city/state mappings |
| `dirty_flights.csv` | 2,376 x 7 | Baran Benchmark | Inconsistent flight codes, scheduled vs. actual arrival drift, flight cancellation sentinels |
| `dirty_rayyan.csv` | 1,000 x 11 | Baran Benchmark | Academic paper metadata, inconsistent author string formats, journal title variations |
| `fifa21_raw_sample.csv` | 10,000 x 77 | Kaggle (yagunana) | Newline character pollution in player names, contract year ranges, wage suffixes (`€150K`, `€1.2M`) |
| `retail_store_sales.csv` | 10,000 x 6 | Kaggle Sales | Missing categories, negative transaction quantities, string-encoded date formatting discrepancies |
| `unclean_data.csv` | 14 x 5 | Kaggle | Tiny edge-case dataset with extreme missingness, multi-type column corruption, and trailing spaces |
| `uncleaned_ds_jobs.csv` | 672 x 15 | Kaggle (Glassdoor) | Salary estimates encoded as text ranges (`$111K-$181K (Glassdoor est.)`), competitor noise |
| `healthcare_messy.csv` | 1,000 x 15 | Synthetic Healthcare | Blood pressure string notation, admission dates after discharge dates, medical condition typos |
| `messy_hr.csv` | 311 x 36 | Kaggle HR | Multi-format hire dates, salary variances, ambiguous performance rating strings |
| `warehouse_messy.csv` | 1,000 x 12 | Supply Chain | Inventory stock count negatives, warehouse zone casing discrepancies, SKU code variations |
| `automobile_messy.csv` | 205 x 26 | UCI Automotive | Disguised `?` missing sentinels in numeric normalized losses and bore/stroke columns |
| `cars93_miss.csv` | 93 x 27 | MASS Package | Classical missing value pattern benchmark across vehicle specifications |
| `cucum_dirty.csv` | 100 x 6 | Synthetic Agronomy | Outlier cucumber dimensions, unit mismatches (cm vs. inches) |
| `planets_messy.csv` | 1,035 x 6 | Seaborn / Exoplanet | High missingness in orbital periods and planet masses, multi-modal observational bias |
| `gapminder_unclean.csv` | 1,704 x 6 | Gapminder Foundation | Country name variants, life expectancy outliers, string-encoded GDP per capita |
| `salaries_dirty_sample.csv` | 10,000 x 11 | SF OpenData | Employee benefit calculations, job classification noise, disguised zero earnings |
| `traffic_dirty1.csv` | 100 x 8 | Sensor Telemetry | Traffic volume sensor dropouts, timestamp irregularities |
| `traffic_dirty2.csv` | 100 x 8 | Sensor Telemetry | Sensor baseline drift, duplicate reading timestamps |

---

## 3. Real-World Datasets (`data/real_world/`)

Diverse multi-domain tabular datasets representing enterprise and production scenarios.

| File | Rows x Cols | Domain | Target Column | Usage |
| :--- | :--- | :--- | :--- | :--- |
| `adult_census.csv` | 32,561 x 15 | Demographic / Socioeconomic | `14` (Income >50K) | Full-Scale Benchmark & Classification Validation |
| `airbnb_nyc.csv` | 48,895 x 16 | Hospitality / Real Estate | `price` | High-cardinality neighborhood names, text descriptions, price outliers |
| `telco_corrupted.csv` | 7,043 x 21 | Telecom Churn | `Churn` | Injected synthetic corruptions with exact ground-truth manifest |
| `netflix_titles.csv` | 8,807 x 12 | Entertainment Media | `type` | Multi-valued cast strings, missing director metadata, country variations |
| `bank_marketing.csv` | 45,211 x 17 | Banking / Financial | `y` | Unknown client sentinels, duration skewness, campaign frequency |
| `car_dekho.csv` | 301 x 9 | Automotive Sales | `Selling_Price` | Showroom prices, fuel types, seller types, transmission categories |
| `customer_churn.csv` | 50 x 8 | Customer Support | `Churn` | Small-sample validation for human review and invariant verification |
| `imdb_top1000.csv` | 1,000 x 16 | Film Industry | `IMDB_Rating` | Runtime string parsing, certificate categorizations, gross revenues |
| `student_performance.csv` | 395 x 33 | Academic Education | `G3` | Student grades, binary attendance flags, parent education levels |
| `breast_cancer.csv` | 569 x 32 | Healthcare / Oncology | `diagnosis` | Continuous biometric features, scaling variance, extreme precision |
| `world_happiness.csv` | 156 x 9 | Macroeconomic Metrics | `Score` | Multi-factor index rankings, national GDP per capita, life expectancy |

---

## 4. Reference ML Datasets (`data/downloaded/`)

Classical machine learning benchmarks used for regression and classification baselines.

| File | Rows x Cols | Task | Target Column |
| :--- | :--- | :--- | :--- |
| `california_housing.csv` | 20,640 x 9 | Regression | `median_house_value` |
| `titanic.csv` | 891 x 12 | Classification | `Survived` |
| `winequality_red.csv` | 1,599 x 12 | Classification / Regression | `quality` |
| `abalone.csv` | 4,177 x 9 | Regression | `Rings` |
| `auto_mpg.csv` | 398 x 9 | Regression | `mpg` |
| `boston_housing.csv` | 506 x 14 | Regression | `MEDV` |
| `diabetes.csv` | 768 x 9 | Classification | `Outcome` |
| `heart_disease.csv` | 303 x 14 | Classification | `target` |
| `penguins.csv` | 344 x 7 | Classification | `species` |
| `iris.csv` | 150 x 5 | Classification | `species` |
