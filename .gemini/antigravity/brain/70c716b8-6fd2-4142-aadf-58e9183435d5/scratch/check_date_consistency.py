import pandas as pd

# Load df dates for a single geocode
df = pd.read_csv("final_brazil_dengue.csv", usecols=["date", "geocode"])
df_single = df[df["geocode"] == 1100015].copy()
df_single["date"] = pd.to_datetime(df_single["date"])
df_single = df_single.sort_values("date")

print("Number of rows for geocode 1100015:", len(df_single))
print("Diffs between consecutive dates:")
print(df_single["date"].diff().value_counts())

# Check if other municipalities have the same row count
counts = df["geocode"].value_counts()
print("\nUnique row counts per geocode:")
print(counts.value_counts())
