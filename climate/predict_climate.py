import os
import gc
import warnings
import datetime
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def get_epiweek_from_sunday(dt):
    year = dt.year
    
    def get_epiweek_1_sunday(y):
        first_jan = datetime.date(y, 1, 1)
        first_jan_wd = first_jan.weekday()  # 0=Monday, ..., 6=Sunday
        days_to_wed = (2 - first_jan_wd) % 7
        first_wed = first_jan + datetime.timedelta(days=days_to_wed)
        epiweek_1_sun = first_wed - datetime.timedelta(days=3)
        return pd.Timestamp(epiweek_1_sun)
    
    sun_this_year = get_epiweek_1_sunday(year)
    if dt >= sun_this_year:
        sun_next_year = get_epiweek_1_sunday(year + 1)
        if dt >= sun_next_year:
            epi_year = year + 1
            week_num = 1 + int((dt - sun_next_year).days / 7)
        else:
            epi_year = year
            week_num = 1 + int((dt - sun_this_year).days / 7)
    else:
        sun_prev_year = get_epiweek_1_sunday(year - 1)
        epi_year = year - 1
        week_num = 1 + int((dt - sun_prev_year).days / 7)
        
    return epi_year, week_num

def predict_climate():
    print("Starting Climate Forecasting (2025-2029)...")
    
    csv_path = "final_brazil_dengue.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "final_brazil_dengue.csv")
        
    targets = [
        "temp_min", "temp_med", "temp_max",
        "precip_med", "precip_tot", "pressure_med",
        "rel_humid_med", "thermal_range", "rainy_days"
    ]
    
    # 1. Load historical data to construct metadata and initial history cache
    print("Loading historical data metadata...")
    df = pd.read_csv(csv_path, usecols=["date", "geocode", "uf", "climate_zone", "epiweek"] + targets)
    df = df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    
    geocodes = df["geocode"].unique()
    num_muns = len(geocodes)
    print(f"Number of municipalities: {num_muns}")
    
    # Extract metadata for geocodes
    meta_df = df[["geocode", "uf", "climate_zone"]].drop_duplicates(subset=["geocode"]).set_index("geocode").reindex(geocodes)
    
    # Get last 52 dates in historical data (ending 2024-06-02)
    last_52_dates = sorted(df["date"].unique())[-52:]
    print(f"History cache start date: {last_52_dates[0].strftime('%Y-%m-%d')}, end date: {last_52_dates[-1].strftime('%Y-%m-%d')}")
    
    # Generate weekly dates for the forecast period (2024-06-09 to 2029-12-30, 290 weeks)
    start_forecast_date = last_52_dates[-1] + pd.Timedelta(days=7)
    forecast_dates = pd.date_range(start=start_forecast_date, end="2029-12-30", freq="7D")
    num_weeks = len(forecast_dates)
    print(f"Forecast weeks: {num_weeks} (from {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')})")
    
    # Precompute time features for forecast weeks
    week_seq = []
    month_seq = []
    sin_week_seq = []
    cos_week_seq = []
    forecast_years = []
    forecast_weeks = []
    
    for dt in forecast_dates:
        epi_yr, epi_wk = get_epiweek_from_sunday(dt)
        week_seq.append(epi_wk)
        month_seq.append(dt.month)
        sin_week_seq.append(np.sin(2 * np.pi * epi_wk / 52.17857))
        cos_week_seq.append(np.cos(2 * np.pi * epi_wk / 52.17857))
        forecast_years.append(epi_yr)
        forecast_weeks.append(epi_wk)
        
    forecasts = {}
    
    # Predict recursively target-by-target
    for target in targets:
        print(f"\nForecasting target: {target}...")
        model_path = f"climate/models/climate_{target}.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Train models first.")
            
        model = joblib.load(model_path)
        
        # Prepare history array: shape (num_muns, 52 + num_weeks)
        # first 52 columns are initialized with historical data
        history = np.zeros((num_muns, 52 + num_weeks), dtype=np.float32)
        
        pivot_df = df[df["date"].isin(last_52_dates)].pivot(index="geocode", columns="date", values=target)
        pivot_df = pivot_df.ffill(axis=1).bfill(axis=1)
        pivot_df = pivot_df.reindex(geocodes)
        
        history[:, :52] = pivot_df.values
        
        # Precompute historical weekly baseline for target
        print("Computing historical weekly baseline...")
        df_hist = df.copy()
        df_hist["week"] = (df_hist["epiweek"] % 100).astype(np.int16)
        df_train = df_hist[df_hist["date"].dt.year <= 2021]
        hist_mean_df = df_train.groupby(["geocode", "week"])[target].mean().reset_index().rename(columns={target: f"hist_mean_{target}"})
        
        hist_mean_pivot = hist_mean_df.pivot(index="geocode", columns="week", values=f"hist_mean_{target}").reindex(geocodes)
        hist_mean_pivot = hist_mean_pivot.ffill(axis=1).bfill(axis=1)
        overall_week_mean = df_train.groupby("week")[target].mean().reindex(range(1, 54)).ffill().bfill().values
        for col in hist_mean_pivot.columns:
            hist_mean_pivot[col] = hist_mean_pivot[col].fillna(overall_week_mean[col - 1])
        hist_mean_matrix = hist_mean_pivot.values  # shape (num_muns, 53)
        
        del df_hist, df_train, hist_mean_df, hist_mean_pivot
        gc.collect()
        
        # Recursive loop over weeks
        for t in range(num_weeks):
            # Extract historical baseline for this week
            wk_val = week_seq[t]
            hist_mean_val = hist_mean_matrix[:, wk_val - 1]
            
            # Construct features
            X_t = np.column_stack([
                # Lags
                history[:, 52 + t - 1],
                history[:, 52 + t - 2],
                history[:, 52 + t - 3],
                history[:, 52 + t - 4],
                history[:, 52 + t - 8],
                history[:, 52 + t - 12],
                history[:, 52 + t - 16],
                history[:, 52 + t - 20],
                history[:, 52 + t - 26],
                history[:, 52 + t - 39],
                history[:, 52 + t - 52],
                # Rolling Means
                history[:, 52 + t - 4 : 52 + t].mean(axis=1),
                history[:, 52 + t - 8 : 52 + t].mean(axis=1),
                history[:, 52 + t - 12 : 52 + t].mean(axis=1),
                history[:, 52 + t - 26 : 52 + t].mean(axis=1),
                history[:, 52 + t - 52 : 52 + t].mean(axis=1),
                # Rolling Stds
                history[:, 52 + t - 4 : 52 + t].std(axis=1),
                history[:, 52 + t - 8 : 52 + t].std(axis=1),
                history[:, 52 + t - 12 : 52 + t].std(axis=1),
                history[:, 52 + t - 26 : 52 + t].std(axis=1),
                history[:, 52 + t - 52 : 52 + t].std(axis=1),
                # Time features
                np.full(num_muns, week_seq[t], dtype=np.float32),
                np.full(num_muns, month_seq[t], dtype=np.float32),
                np.full(num_muns, sin_week_seq[t], dtype=np.float32),
                np.full(num_muns, cos_week_seq[t], dtype=np.float32),
                # Historical baseline prior
                hist_mean_val
            ])
            
            # Predict for all municipalities
            preds = model.predict(X_t)
            history[:, 52 + t] = preds
            
        forecasts[target] = history[:, 52:]
        del model, history, pivot_df
        gc.collect()
        
    print("\nAssembling forecast dataframe...")
    # Assemble long-format dataframe
    week_dfs = []
    for t in range(num_weeks):
        wk_df = pd.DataFrame(index=geocodes)
        wk_df["date"] = forecast_dates[t].strftime("%Y-%m-%d")
        wk_df["year"] = forecast_years[t]
        wk_df["week"] = forecast_weeks[t]
        wk_df["geocode"] = geocodes
        wk_df["uf"] = meta_df["uf"].values
        wk_df["climate_zone"] = meta_df["climate_zone"].values
        
        for target in targets:
            wk_df[target] = forecasts[target][:, t]
            
        week_dfs.append(wk_df)
        
    forecast_df = pd.concat(week_dfs, ignore_index=True)
    
    # Keep only 2025-2029 forecast as requested
    forecast_2025_2029 = forecast_df[(forecast_df["year"] >= 2025) & (forecast_df["year"] <= 2029)].reset_index(drop=True)
    
    # Save outputs
    out_path_1 = "climate/outputs/municipality_climate_2025_2029.csv"
    out_path_2 = "final/outputs/csv/municipality_climate_2025_2029.csv"
    out_path_full = "climate/outputs/municipality_climate_2024_2029_full.csv"
    
    forecast_2025_2029.to_csv(out_path_1, index=False)
    forecast_2025_2029.to_csv(out_path_2, index=False)
    forecast_df.to_csv(out_path_full, index=False)
    
    print(f"Saved climate forecasts to {out_path_1} and {out_path_2}")
    print(f"Saved full forecast to {out_path_full}")
    print(f"Forecast data shape: {forecast_2025_2029.shape}")

if __name__ == "__main__":
    predict_climate()
