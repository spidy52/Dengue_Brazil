import pandas as pd
import datetime

def get_epiweek_from_sunday(dt):
    # dt is a pandas Timestamp representing a Sunday
    year = dt.year
    
    # Function to get the Sunday of epiweek 1 for a given year
    def get_epiweek_1_sunday(y):
        # Find the first Wednesday of January of year y
        first_jan = datetime.date(y, 1, 1)
        # weekday: 0=Monday, ..., 2=Wednesday, ..., 6=Sunday
        first_jan_wd = first_jan.weekday()
        # Days to Wednesday
        days_to_wed = (2 - first_jan_wd) % 7
        first_wed = first_jan + datetime.timedelta(days=days_to_wed)
        # Sunday of that week (which is the Sunday before or equal to the Wednesday)
        # Since Sunday is 6, if Wednesday is 2, the Sunday is 3 days before Wednesday.
        epiweek_1_sun = first_wed - datetime.timedelta(days=3)
        return pd.Timestamp(epiweek_1_sun)
    
    sun_this_year = get_epiweek_1_sunday(year)
    if dt >= sun_this_year:
        # Check if it belongs to next year's epiweek 1
        sun_next_year = get_epiweek_1_sunday(year + 1)
        if dt >= sun_next_year:
            epi_year = year + 1
            week_num = 1 + int((dt - sun_next_year).days / 7)
        else:
            epi_year = year
            week_num = 1 + int((dt - sun_this_year).days / 7)
    else:
        # Belongs to previous year
        sun_prev_year = get_epiweek_1_sunday(year - 1)
        epi_year = year - 1
        week_num = 1 + int((dt - sun_prev_year).days / 7)
        
    return epi_year * 100 + week_num

# Test on historical data
df = pd.read_csv("final_brazil_dengue.csv", usecols=["date", "epiweek"]).drop_duplicates().head(5000)
df["date"] = pd.to_datetime(df["date"])
df["calc_epiweek"] = df["date"].apply(get_epiweek_from_sunday)

mismatches = df[df["calc_epiweek"] != df["epiweek"]]
print("Number of mismatches:", len(mismatches))
if len(mismatches) > 0:
    print(mismatches.head(10))
else:
    print("Success! Perfect match on all tested historical rows.")
