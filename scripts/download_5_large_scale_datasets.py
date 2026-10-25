"""
Download 5 Really Really Large Datasets that are Famously Not Clean.
Batch of Massive Real-World Dirty Datasets (1.3+ Million Total Rows).
"""

import os
import io
import urllib.request
import pandas as pd

DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dirty_large_scale")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

DATASETS = {
    "online_retail_540k.csv": {
        "url": "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv",
        "description": "541,909 e-commerce transactions with negative cancelled quantities ('C536379'), 135,000+ missing customer IDs, zero unit prices, and postage outliers",
        "source": "UCI / Databricks",
        "max_rows": None
    },
    "uber_trips_nyc_560k.csv": {
        "url": "https://raw.githubusercontent.com/fivethirtyeight/uber-tlc-foil-response/master/uber-trip-data/uber-raw-data-apr14.csv",
        "description": "564,516 raw NYC ride-hailing trips with unstandardized timestamps, GPS outliers, and dispatch base codes",
        "source": "FiveThirtyEight / NYC TLC",
        "max_rows": None
    },
    "pnw_flights_160k.csv": {
        "url": "https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/flights.csv",
        "description": "162,049 commercial airline flight schedules with missing departure/arrival times, delay sentinels, and airtime corruptions",
        "source": "ismayc / Bureau of Transportation Statistics",
        "max_rows": None
    },
    "sf_payroll_74k.csv": {
        "url": "https://raw.githubusercontent.com/visenger/clean-and-dirty-data/master/SALARIES/salaries_small/salaries-1.csv",
        "description": "74,325 municipal payroll records with 184,800+ missing values, overtime discrepancies, negative base pay, and title casing variants",
        "source": "Raha / Baran Benchmark",
        "max_rows": None
    },
    "chicago_salaries_32k.csv": {
        "url": "https://data.cityofchicago.org/api/views/xzkq-xp2w/rows.csv?accessType=DOWNLOAD",
        "description": "32,192 municipal employee records with employee name formatting variants (Last, First), department acronyms, and mixed hourly/annual rates",
        "source": "City of Chicago Open Data",
        "max_rows": None
    }
}

def download_all():
    print(f"=== DOWNLOADING 5 REALLY LARGE DIRTY DATASETS TO {DOWNLOAD_DIR} ===")
    results = {}
    for filename, info in DATASETS.items():
        dest = os.path.join(DOWNLOAD_DIR, filename)
        url = info["url"]
        print(f"\nFetching {filename} from {info['source']}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = resp.read()
            df = pd.read_csv(io.BytesIO(content))
            
            if info.get("max_rows") and len(df) > info["max_rows"]:
                print(f"  Sampling {info['max_rows']} rows from {len(df)}...")
                df = df.iloc[:info["max_rows"]].copy()

            df.to_csv(dest, index=False)
            results[filename] = {
                "status": "SUCCESS",
                "rows": len(df),
                "cols": len(df.columns),
                "null_count": int(df.isna().sum().sum()),
                "path": dest,
                "description": info["description"]
            }
            print(f"  [OK] Saved {filename} ({len(df):,} rows x {len(df.columns)} cols, {results[filename]['null_count']:,} nulls)")
        except Exception as e:
            print(f"  [FAIL] Error downloading {filename}: {e}")
            results[filename] = {"status": "FAILED", "error": str(e)}

    return results

if __name__ == "__main__":
    download_all()
