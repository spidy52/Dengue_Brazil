import pandas as pd

# Load date and geocode
df = pd.read_csv("final_brazil_dengue.csv", usecols=["date", "geocode"])
duplicates = df.duplicated(subset=["date", "geocode"])
print("Total duplicates (same date and geocode):", duplicates.sum())

if duplicates.sum() > 0:
    print("\nSample duplicates:")
    dup_rows = df[df.duplicated(subset=["date", "geocode"], keep=False)].sort_values(["geocode", "date"]).head(10)
    print(dup_rows)
