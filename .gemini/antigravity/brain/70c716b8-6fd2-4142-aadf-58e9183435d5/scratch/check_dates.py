import pandas as pd

# Load df columns of interest
df = pd.read_csv("final_brazil_dengue.csv", usecols=["date", "year", "geocode"])
print("Total rows:", len(df))
print("Unique years in dataset:", df["year"].unique())

# Group by geocode and get the max date for a few
print("\nMax dates for some municipalities:")
print(df.groupby("geocode")["date"].max().head(10))

print("\nMin dates for some municipalities:")
print(df.groupby("geocode")["date"].min().head(10))

# Count of rows per year
print("\nRow counts per year:")
print(df["year"].value_counts().sort_index())
