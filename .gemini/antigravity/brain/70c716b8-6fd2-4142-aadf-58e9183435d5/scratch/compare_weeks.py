import pandas as pd

df = pd.read_csv("final_brazil_dengue.csv", nrows=10000, usecols=["date", "epiweek"])
df["date"] = pd.to_datetime(df["date"])
df["iso_week"] = df["date"].dt.isocalendar().week
df["epiweek_num"] = df["epiweek"] % 100

# Compare
mismatches = df[df["iso_week"] != df["epiweek_num"]]
print("Number of mismatches in first 10,000 rows:", len(mismatches))
if len(mismatches) > 0:
    print("\nSample mismatches:")
    print(mismatches.head(10))
