import pandas as pd

# Load duplicate rows for a single geocode
df = pd.read_csv("final_brazil_dengue.csv")
dup_mask = df.duplicated(subset=["date", "geocode"], keep=False)
df_dup = df[dup_mask].sort_values(["geocode", "date"])

# Print first few duplicate rows including all columns
print("Sample duplicates with columns:")
print(df_dup.head(4).to_string())

# Check if they are identical across all columns
all_dup = df.duplicated()
print("\nNumber of exact duplicates across all columns:", all_dup.sum())
