import pandas as pd
import numpy as np

# Load a sample of the data to inspect columns and data types
df_sample = pd.read_csv("final_brazil_dengue.csv", nrows=100)
print("Columns:")
print(df_sample.columns.tolist())
print("\nDtypes:")
print(df_sample.dtypes)

# Check unique climate zones and unique states (uf)
df_full_cols = pd.read_csv("final_brazil_dengue.csv", usecols=["date", "climate_zone", "uf", "geocode"])
print("\nDate range:")
print("Min:", df_full_cols["date"].min())
print("Max:", df_full_cols["date"].max())

print("\nUnique climate zones:")
print(df_full_cols["climate_zone"].unique())

print("\nNumber of unique municipalities:")
print(df_full_cols["geocode"].nunique())

print("\nNumber of unique states (uf):")
print(df_full_cols["uf"].nunique())
print(df_full_cols["uf"].unique())
