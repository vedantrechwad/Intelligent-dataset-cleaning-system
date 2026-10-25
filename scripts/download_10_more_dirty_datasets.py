"""
Download 10 More Actually Dirty Datasets from Internet Benchmark Repositories.
Batch 2 of Real-World Dirty Data Evaluation.
"""

import os
import io
import urllib.request
import pandas as pd

DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dirty_downloads_round2")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

DATASETS = {
    "dirty_cafe_sales.csv": {
        "url": "https://raw.githubusercontent.com/shaadyalii/Dirty-Cafe-Sales-Dataset/main/dirty_cafe_sales.csv",
        "description": "Cafe sales transactions with missing quantities, unit prices, corrupted UNKNOWN/ERROR tokens",
        "source": "shaadyalii/Dirty-Cafe-Sales-Dataset"
    },
    "retail_store_sales.csv": {
        "url": "https://raw.githubusercontent.com/AungHtet2/Retail-Store-Sales-Dirty-for-Data-Cleaning/main/csv/retail_store_sales.csv",
        "description": "Retail store transactions with corrupted item categories, prices, negative quantities, missing discounts",
        "source": "AungHtet2/Retail-Store-Sales-Dirty-for-Data-Cleaning"
    },
    "beers_dirty.csv": {
        "url": "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/beers/dirty.csv",
        "description": "Classic Baran data cleaning benchmark for craft beers with missing ABVs, ounces, and brewery typos",
        "source": "BigDaMa/raha"
    },
    "movies_dirty.csv": {
        "url": "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/movies_1/dirty.csv",
        "description": "Baran movie metadata benchmark with corrupt release dates, director misalignments, and missing genres",
        "source": "BigDaMa/raha"
    },
    "healthcare_messy.csv": {
        "url": "https://raw.githubusercontent.com/eyowhite/Messy-dataset/main/healthcare_messy_data.csv",
        "description": "Hospital patient intake records with corrupted ages, typos in medical conditions, and inconsistent dosages",
        "source": "eyowhite/Messy-dataset"
    },
    "messy_hr.csv": {
        "url": "https://raw.githubusercontent.com/eyowhite/Messy-dataset/main/messy_HR_data.csv",
        "description": "Corporate HR census with salary format discrepancies, department typos, and missing gender records",
        "source": "eyowhite/Messy-dataset"
    },
    "warehouse_messy.csv": {
        "url": "https://raw.githubusercontent.com/eyowhite/Messy-dataset/main/warehouse_messy_data.csv",
        "description": "Supply chain warehouse inventory with invalid product IDs, missing categories, and location discrepancies",
        "source": "eyowhite/Messy-dataset"
    },
    "automobile_messy.csv": {
        "url": "https://raw.githubusercontent.com/eyowhite/Messy-dataset/main/automobile_dataset.csv",
        "description": "Automotive specifications with sentinel values ('?'), missing prices, normalized losses, and horsepower nulls",
        "source": "eyowhite/Messy-dataset"
    },
    "traffic_dirty1.csv": {
        "url": "https://raw.githubusercontent.com/kevinmcaleer/data_cleansing/main/dirty_data1.csv",
        "description": "Traffic survey records with dirty car colors, invalid dates, and inconsistent occupants",
        "source": "kevinmcaleer/data_cleansing"
    },
    "traffic_dirty2.csv": {
        "url": "https://raw.githubusercontent.com/kevinmcaleer/data_cleansing/main/dirty_data2.csv",
        "description": "Second-stage traffic survey records with relational dependency violations across make, model, and occupants",
        "source": "kevinmcaleer/data_cleansing"
    }
}

def download_all():
    print(f"=== DOWNLOADING 10 NEW DIRTY DATASETS TO {DOWNLOAD_DIR} ===")
    results = {}
    for filename, info in DATASETS.items():
        dest = os.path.join(DOWNLOAD_DIR, filename)
        url = info["url"]
        print(f"\nFetching {filename} from {info['source']}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
            df = pd.read_csv(io.BytesIO(content))
            df.to_csv(dest, index=False)
            results[filename] = {
                "status": "SUCCESS",
                "rows": len(df),
                "cols": len(df.columns),
                "null_count": int(df.isna().sum().sum()),
                "path": dest,
                "description": info["description"]
            }
            print(f"  [OK] Saved {filename} ({len(df)} rows x {len(df.columns)} cols, {results[filename]['null_count']} nulls)")
        except Exception as e:
            print(f"  [FAIL] Error downloading {filename}: {e}")
            results[filename] = {"status": "FAILED", "error": str(e)}

    return results

if __name__ == "__main__":
    download_all()
