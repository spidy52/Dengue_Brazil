import pandas as pd
import numpy as np
import time
import gc

start_time = time.time()

# Load only date, geocode, and temp_med to test
print("Loading data...")
df = pd.read_csv("final_brazil_dengue.csv", usecols=["date", "geocode", "temp_med"])
print(f"Loaded in {time.time() - start_time:.2f} seconds")

# Drop duplicates
df = df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
print(f"Deduplicated and sorted. Rows: {len(df)}")

# Benchmarking feature engineering
print("Generating features...")
t0 = time.time()

# Lags
lags = [1, 2, 3, 4, 8, 12, 16, 20, 26, 39, 52]
for lag in lags:
    df[f"lag_{lag}"] = df.groupby("geocode")["temp_med"].shift(lag).astype(np.float32)

print(f"Lags created in {time.time() - t0:.2f} seconds")

# Rolling Mean
t0 = time.time()
windows = [4, 8, 12, 26, 52]
for w in windows:
    df[f"roll_mean_{w}"] = df.groupby("geocode")["temp_med"].shift(1).rolling(w).mean().astype(np.float32)
print(f"Rolling means created in {time.time() - t0:.2f} seconds")

# Rolling Std
t0 = time.time()
for w in windows:
    df[f"roll_std_{w}"] = df.groupby("geocode")["temp_med"].shift(1).rolling(w).std().astype(np.float32)
print(f"Rolling stds created in {time.time() - t0:.2f} seconds")

print(f"Total time: {time.time() - start_time:.2f} seconds")
print(df.info(memory_usage="deep"))
