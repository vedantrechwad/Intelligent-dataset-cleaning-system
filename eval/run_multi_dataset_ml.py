import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine

def get_cleaned(dataset_name):
    dirty_path = os.path.join(WORKSPACE_ROOT, "benchmarks", dataset_name, "dirty.csv")
    df_dirty = load_table(dirty_path)
    prof = profile_table(df_dirty)
    engine = CleaningEngine()
    props = engine.generate_proposals(df_dirty, prof)
    approved = [p for p in props if p.tier in ["AUTO", "REVIEW"]]
    df_out, _ = engine.apply(df_dirty, approved)
    return df_dirty, df_out

print("=======================================================================", flush=True)
print("MULTI-DATASET DOWNSTREAM MACHINE LEARNING BENCHMARKS", flush=True)
print("=======================================================================", flush=True)

# ----------------------------------------------------
# 1. TELCO (Classification: Target = Churn)
# ----------------------------------------------------
print("\n[1/3] Running Downstream ML on TELCO (Target: Churn [Classification])...", flush=True)
df_d_telco, df_o_telco = get_cleaned("telco")
df_c_telco = load_table(os.path.join(WORKSPACE_ROOT, "benchmarks", "telco", "clean.csv"))

def prep_telco(df):
    X = df.drop(columns=["Churn", "customerID"], errors="ignore").copy()
    y = df["Churn"].str.strip().str.capitalize().map({"Yes": 1, "No": 0}).fillna(0)
    for c in X.columns:
        num = pd.to_numeric(X[c], errors="coerce")
        if num.notna().sum() > len(X) * 0.5:
            X[c] = num.fillna(num.median())
        else:
            X[c] = X[c].astype(str)
    return pd.get_dummies(X, drop_first=True), y

X_d_tel, y_d_tel = prep_telco(df_d_telco)
X_o_tel, y_o_tel = prep_telco(df_o_telco)
X_c_tel, y_c_tel = prep_telco(df_c_telco)

common_cols = list(set(X_d_tel.columns).intersection(set(X_o_tel.columns)).intersection(set(X_c_tel.columns)))
X_d_tel, X_o_tel, X_c_tel = X_d_tel[common_cols], X_o_tel[common_cols], X_c_tel[common_cols]

idx = list(range(len(df_c_telco)))
tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=42)
X_test_tel, y_test_tel = X_c_tel.iloc[te_idx], y_c_tel.iloc[te_idx]

m_d_tel = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_d_tel.iloc[tr_idx], y_d_tel.iloc[tr_idx])
m_o_tel = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_o_tel.iloc[tr_idx], y_o_tel.iloc[tr_idx])
m_c_tel = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_c_tel.iloc[tr_idx], y_c_tel.iloc[tr_idx])

p_d_tel = m_d_tel.predict(X_test_tel)
p_o_tel = m_o_tel.predict(X_test_tel)
p_c_tel = m_c_tel.predict(X_test_tel)

telco_res = {
    "Dirty": {"Acc": accuracy_score(y_test_tel, p_d_tel), "F1": f1_score(y_test_tel, p_d_tel)},
    "Our Clean": {"Acc": accuracy_score(y_test_tel, p_o_tel), "F1": f1_score(y_test_tel, p_o_tel)},
    "Clean GT": {"Acc": accuracy_score(y_test_tel, p_c_tel), "F1": f1_score(y_test_tel, p_c_tel)}
}
print(f"  -> Telco Done: Dirty F1={telco_res['Dirty']['F1']*100:.2f}%, Our F1={telco_res['Our Clean']['F1']*100:.2f}%, Clean GT F1={telco_res['Clean GT']['F1']*100:.2f}%", flush=True)

# ----------------------------------------------------
# 2. BEERS (Regression: Target = abv)
# ----------------------------------------------------
print("\n[2/3] Running Downstream ML on BEERS (Target: abv [Regression])...", flush=True)
df_d_beer, df_o_beer = get_cleaned("beers")
df_c_beer = load_table(os.path.join(WORKSPACE_ROOT, "benchmarks", "beers", "clean.csv"))

def prep_beer(df, target_y=None):
    X = df.drop(columns=["abv", "index", "id", "beer-name"], errors="ignore").copy()
    if target_y is None:
        y = pd.to_numeric(df["abv"].str.replace("%", "").str.strip(), errors="coerce").fillna(0.05)
    else:
        y = target_y
    for c in X.columns:
        num = pd.to_numeric(X[c].str.extract(r'(\d+\.?\d*)')[0], errors="coerce")
        if num.notna().sum() > len(X) * 0.4:
            X[c] = num.fillna(num.median())
        else:
            # Low cardinality category
            X[c] = X[c].astype(str)
    return pd.get_dummies(X, drop_first=True), y

y_ground_beer = pd.to_numeric(df_c_beer["abv"].str.replace("%", "").str.strip(), errors="coerce").fillna(0.05)
X_d_b, y_d_b = prep_beer(df_d_beer)
X_o_b, y_o_b = prep_beer(df_o_beer)
X_c_b, y_c_b = prep_beer(df_c_beer, target_y=y_ground_beer)

common_b = list(set(X_d_b.columns).intersection(set(X_o_b.columns)).intersection(set(X_c_b.columns)))
X_d_b, X_o_b, X_c_b = X_d_b[common_b], X_o_b[common_b], X_c_b[common_b]

idx_b = list(range(len(df_c_beer)))
tr_idx_b, te_idx_b = train_test_split(idx_b, test_size=0.3, random_state=42)
X_test_b, y_test_b = X_c_b.iloc[te_idx_b], y_ground_beer.iloc[te_idx_b]

m_d_b = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_d_b.iloc[tr_idx_b], y_d_b.iloc[tr_idx_b])
m_o_b = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_o_b.iloc[tr_idx_b], y_o_b.iloc[tr_idx_b])
m_c_b = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_c_b.iloc[tr_idx_b], y_c_b.iloc[tr_idx_b])

p_d_b = m_d_b.predict(X_test_b)
p_o_b = m_o_b.predict(X_test_b)
p_c_b = m_c_b.predict(X_test_b)

beers_res = {
    "Dirty": {"R2": r2_score(y_test_b, p_d_b), "RMSE": np.sqrt(mean_squared_error(y_test_b, p_d_b))},
    "Our Clean": {"R2": r2_score(y_test_b, p_o_b), "RMSE": np.sqrt(mean_squared_error(y_test_b, p_o_b))},
    "Clean GT": {"R2": r2_score(y_test_b, p_c_b), "RMSE": np.sqrt(mean_squared_error(y_test_b, p_c_b))}
}
print(f"  -> Beers Done: Dirty R2={beers_res['Dirty']['R2']:.3f}, Our R2={beers_res['Our Clean']['R2']:.3f}, Clean GT R2={beers_res['Clean GT']['R2']:.3f}", flush=True)

# ----------------------------------------------------
# 3. MOVIES_1 (Regression: Target = RatingValue)
# ----------------------------------------------------
print("\n[3/3] Running Downstream ML on MOVIES_1 (Target: RatingValue [Regression])...", flush=True)
df_d_mov, df_o_mov = get_cleaned("movies_1")
df_c_mov = load_table(os.path.join(WORKSPACE_ROOT, "benchmarks", "movies_1", "clean.csv"))

# Subsample 4000 rows for fast execution
df_c_mov_sub = df_c_mov.iloc[:4000]
df_d_mov_sub = df_d_mov.iloc[:4000]
df_o_mov_sub = df_o_mov.iloc[:4000]

def prep_mov(df):
    df_copy = df.copy()
    df_copy.columns = [c.lower().replace(" ", "").replace("_", "") for c in df_copy.columns]
    X = df_copy[["duration", "year", "ratingcount", "reviewcount", "genre"]].copy()
    y = pd.to_numeric(df_copy["ratingvalue"], errors="coerce").fillna(6.0)
    for c in ["duration", "year", "ratingcount", "reviewcount"]:
        num = pd.to_numeric(X[c].astype(str).str.extract(r'(\d+)')[0], errors="coerce")
        X[c] = num.fillna(num.median() if pd.notna(num.median()) else 0)
    X = pd.get_dummies(X, columns=["genre"], drop_first=True)
    return X, y

X_d_m, y_d_m = prep_mov(df_d_mov_sub)
X_o_m, y_o_m = prep_mov(df_o_mov_sub)
X_c_m, y_c_m = prep_mov(df_c_mov_sub)

common_m = list(set(X_d_m.columns).intersection(set(X_o_m.columns)).intersection(set(X_c_m.columns)))
X_d_m, X_o_m, X_c_m = X_d_m[common_m], X_o_m[common_m], X_c_m[common_m]

idx_m = list(range(len(df_c_mov_sub)))
tr_idx_m, te_idx_m = train_test_split(idx_m, test_size=0.3, random_state=42)
X_test_m, y_test_m = X_c_m.iloc[te_idx_m], y_c_m.iloc[te_idx_m]

m_d_m = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_d_m.iloc[tr_idx_m], y_d_m.iloc[tr_idx_m])
m_o_m = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_o_m.iloc[tr_idx_m], y_o_m.iloc[tr_idx_m])
m_c_m = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_c_m.iloc[tr_idx_m], y_c_m.iloc[tr_idx_m])

p_d_m = m_d_m.predict(X_test_m)
p_o_m = m_o_m.predict(X_test_m)
p_c_m = m_c_m.predict(X_test_m)

movies_res = {
    "Dirty": {"R2": r2_score(y_test_m, p_d_m), "RMSE": np.sqrt(mean_squared_error(y_test_m, p_d_m))},
    "Our Clean": {"R2": r2_score(y_test_m, p_o_m), "RMSE": np.sqrt(mean_squared_error(y_test_m, p_o_m))},
    "Clean GT": {"R2": r2_score(y_test_m, p_c_m), "RMSE": np.sqrt(mean_squared_error(y_test_m, p_c_m))}
}
print(f"  -> Movies Done: Dirty R2={movies_res['Dirty']['R2']:.3f}, Our R2={movies_res['Our Clean']['R2']:.3f}, Clean GT R2={movies_res['Clean GT']['R2']:.3f}", flush=True)

print("\n" + "=" * 70, flush=True)
print("MULTI-DATASET MACHINE LEARNING BENCHMARK SUMMARY TABLE", flush=True)
print("=======================================================================", flush=True)
print("{:<12} {:<15} {:<18} {:<18} {:<18}".format("Dataset", "ML Task", "Dirty Dataset", "Our Cleaned", "Clean GT (Provided)"))
print("-" * 75)
print("{:<12} {:<15} {:<18} {:<18} {:<18}".format(
    "Telco", "Classification", 
    f"F1: {telco_res['Dirty']['F1']*100:.2f}%", 
    f"F1: {telco_res['Our Clean']['F1']*100:.2f}%", 
    f"F1: {telco_res['Clean GT']['F1']*100:.2f}%"
))
print("{:<12} {:<15} {:<18} {:<18} {:<18}".format(
    "Beers", "Regression", 
    f"R2: {beers_res['Dirty']['R2']:.3f}", 
    f"R2: {beers_res['Our Clean']['R2']:.3f}", 
    f"R2: {beers_res['Clean GT']['R2']:.3f}"
))
print("{:<12} {:<15} {:<18} {:<18} {:<18}".format(
    "Movies_1", "Regression", 
    f"R2: {movies_res['Dirty']['R2']:.3f}", 
    f"R2: {movies_res['Our Clean']['R2']:.3f}", 
    f"R2: {movies_res['Clean GT']['R2']:.3f}"
))
print("=" * 75, flush=True)
