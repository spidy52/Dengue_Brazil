import pandas as pd

df = pd.read_csv("final_brazil_dengue.csv", usecols=["uf", "geocode", "population", "year"])
# Inspect some years to see if population varies by year or is constant per state
print("Unique populations per state in 2020:")
print(df[df["year"] == 2020].groupby("uf")["population"].unique())

print("\nDoes state population vary by year? Let's check for SP:")
print(df[df["uf"] == "SP"].groupby("year")["population"].unique())
