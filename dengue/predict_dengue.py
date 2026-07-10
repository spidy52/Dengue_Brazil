import os
import gc
import warnings
import datetime
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DYNAMIC_FEATURES = [
    "hist_mean_incidence",
    "population",
    "lag_1_incidence",
    "diff_lag_1",
    "lag_2_incidence",
    "lag_52_incidence",
    "lag_104_incidence",
    "cum_incidence_52w",
    "temp_med",
    "temp_anomaly",
    "precip_tot",
    "precip_anomaly"
]

def get_epiweek_from_sunday(dt):
    year = dt.year
    def get_epiweek_1_sunday(y):
        first_jan = datetime.date(y, 1, 1)
        first_jan_wd = first_jan.weekday()
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

def predict_dengue():
    print("Starting Dengue Forecasting using Dynamic LightGBM Model (2025-2029)...")

    hist_csv = "final_brazil_dengue.csv"
    if not os.path.exists(hist_csv):
        hist_csv = os.path.join("data", "final_brazil_dengue.csv")

    climate_forecast_csv = "climate/outputs/municipality_climate_2024_2029_full.csv"
    if not os.path.exists(climate_forecast_csv):
        raise FileNotFoundError(f"Could not find full climate forecast at {climate_forecast_csv}.")

    # Load zone-wise models
    models = {}
    for zone in range(1, 7):
        m_path = f"dengue/models/dengue_model_zone_{zone}.joblib"
        if not os.path.exists(m_path):
            raise FileNotFoundError(f"Zone {zone} model missing at {m_path}. Train model first.")
        models[float(zone)] = joblib.load(m_path)

    residual_path = "dengue/models/residual_info.joblib"
    residual_info = joblib.load(residual_path) if os.path.exists(residual_path) else None
    if residual_info:
        week_residual_std = residual_info["week_std"]
        global_residual_std = residual_info["global_std"]
        print(f"Loaded residual noise info (global_std={global_residual_std:.2f})")
    else:
        week_residual_std = {}
        global_residual_std = 5.0

    # Build a week-indexed noise std array
    noise_std_by_week = np.array([
        week_residual_std.get(w, global_residual_std) for w in range(1, 54)
    ], dtype=np.float32)

    # ---- Stochastic noise scale ----
    # 0.35 means we inject 35% of the historical weekly residual std as random noise.
    # This is enough to create realistic irregular curves without making it pure noise.
    NOISE_SCALE = 0.35
    rng = np.random.default_rng(seed=42)

    # Load historical data
    print("Loading historical data for cache initialization...")
    hist_df = pd.read_csv(hist_csv, usecols=[
        "date", "geocode", "uf", "climate_zone", "population",
        "incidence_rate", "cases", "temp_med", "precip_tot", "rel_humid_med", "epiweek"
    ])
    hist_df = hist_df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    hist_df["date"] = pd.to_datetime(hist_df["date"])

    geocodes = hist_df["geocode"].unique()
    num_muns = len(geocodes)
    pop_series = hist_df.groupby("geocode")["population"].last()
    
    # Correct zone populations: sum of all municipality populations per zone
    zone_pop_dict = (
        hist_df[hist_df["date"].dt.year == 2024]
        .groupby(["geocode","climate_zone"])["population"].first()
        .reset_index()
        .groupby("climate_zone")["population"].sum()
        .to_dict()
    )

    # Load climate predictions
    print("Loading predicted climate variables...")
    clim_pred = pd.read_csv(climate_forecast_csv)
    clim_pred["date"] = pd.to_datetime(clim_pred["date"])
    clim_pred = clim_pred.sort_values(["geocode", "date"]).reset_index(drop=True)

    forecast_dates = sorted(clim_pred["date"].unique())
    num_weeks = len(forecast_dates)
    print(f"Number of forecast weeks: {num_weeks} (from {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')})")

    week_seq = []
    forecast_years = []
    forecast_weeks = []
    for dt in forecast_dates:
        epi_yr, epi_wk = get_epiweek_from_sunday(dt)
        week_seq.append(min(53, max(1, epi_wk)))
        forecast_years.append(epi_yr)
        forecast_weeks.append(min(53, max(1, epi_wk)))

    meta_df = hist_df[["geocode", "uf", "climate_zone"]].drop_duplicates(subset=["geocode"]).set_index("geocode").reindex(geocodes)

    # Compute historical baselines
    print("Computing historical weekly baselines for temperature and precipitation...")
    df_hist = hist_df.copy()
    df_hist["week"] = (df_hist["epiweek"] % 100).astype(np.int16)
    df_train_dengue = df_hist[(df_hist["date"].dt.year >= 2022) & (df_hist["date"].dt.year <= 2024)]
    df_train_climate = df_hist[df_hist["date"].dt.year <= 2021]

    def make_baseline_matrix(df_train, geocodes, col):
        hm = df_train.groupby(["geocode", "week"])[col].mean().reset_index().rename(columns={col: "val"})
        pivot = hm.pivot(index="geocode", columns="week", values="val").reindex(geocodes)
        pivot = pivot.reindex(columns=range(1, 54))
        pivot = pivot.ffill(axis=1).bfill(axis=1)
        overall = df_train.groupby("week")[col].mean().reindex(range(1, 54)).ffill().bfill().values
        for c in pivot.columns:
            pivot[c] = pivot[c].fillna(overall[c - 1])
        return pivot.values.astype(np.float32)

    hist_mean_matrix = make_baseline_matrix(df_train_dengue, geocodes, "incidence_rate")
    
    # ----------------------------------------------------------------
    # Compute zone-specific baseline scales dynamically from hist_df
    # Recent regime baseline is already at correct scale (ratio = 1.0)
    # ----------------------------------------------------------------
    print("Computing dynamic baseline scaling factors per zone...")
    df_hist_p1 = df_train_dengue.copy()
    df_hist_p2 = df_train_dengue.copy()
    
    df_hist_p1["year"] = df_hist_p1["date"].dt.year
    df_hist_p2["year"] = df_hist_p2["date"].dt.year
    
    def get_zone_peak_inc(sub_df):
        if len(sub_df) == 0:
            return {}
        zg = sub_df.groupby(["climate_zone", "date"])["cases"].sum().reset_index()
        zg["population"] = zg["climate_zone"].map(zone_pop_dict)
        zg["incidence_rate"] = (zg["cases"] / zg["population"]) * 100000.0
        return zg.groupby("climate_zone")["incidence_rate"].max().to_dict()
        
    peaks_p1 = get_zone_peak_inc(df_hist_p1)
    peaks_p2 = get_zone_peak_inc(df_hist_p2)
    
    # Map scale factors
    climate_zones = meta_df["climate_zone"].values
    zone_baseline_scales = {}
    for zone in sorted(zone_pop_dict.keys()):
        p1 = peaks_p1.get(zone, 1.0)
        p2 = peaks_p2.get(zone, 1.0)
        ratio = p2 / max(p1, 0.1)
        # Keep scales at least 1.0 (do not scale down) and cap at 15.0 to avoid noise
        zone_baseline_scales[zone] = max(1.0, min(15.0, ratio))
        
    print("Zone baseline scaling factors:")
    for zone, val in zone_baseline_scales.items():
        print(f"  Zone {zone}: {val:.2f}x")
        
    base_scales = np.array([zone_baseline_scales.get(z, 1.0) for z in climate_zones], dtype=np.float32)
    hist_mean_matrix = hist_mean_matrix * base_scales[:, np.newaxis]

    hist_temp_matrix = make_baseline_matrix(df_train_climate, geocodes, "temp_med")
    hist_precip_matrix = make_baseline_matrix(df_train_climate, geocodes, "precip_tot")

    del df_hist, df_train_dengue, df_train_climate
    gc.collect()

    # Initialize 104-week history caches
    last_104_dates = sorted(hist_df["date"].unique())[-104:]
    last_8_dates = sorted(hist_df["date"].unique())[-8:]

    hist_inc_pivot = (
        hist_df[hist_df["date"].isin(last_104_dates)]
        .pivot(index="geocode", columns="date", values="incidence_rate")
        .ffill(axis=1).bfill(axis=1).reindex(geocodes).values.astype(np.float32)
    )
    hist_cases_pivot = (
        hist_df[hist_df["date"].isin(last_104_dates)]
        .pivot(index="geocode", columns="date", values="cases")
        .ffill(axis=1).bfill(axis=1).reindex(geocodes).values.astype(np.float32)
    )

    hist_climate = {}
    for var in ["temp_med", "precip_tot"]:
        hist_climate[var] = (
            hist_df[hist_df["date"].isin(last_8_dates)]
            .pivot(index="geocode", columns="date", values=var)
            .ffill(axis=1).bfill(axis=1).reindex(geocodes).values.astype(np.float32)
        )

    # Reconstruct forecasted climate by combining the smooth predicted climate (from Steps 1 & 2)
    # with the historical weather anomalies (from 2020-2024 shifted by 5 years).
    # This ensures the trend follows the new climate predictions (warming, rainfall changes)
    # while preserving realistic, irregular week-to-week anomalies.
    print("Performing residual anomaly downscaling on forecasted climate variables...")
    hist_cols = hist_df.pivot(index="geocode", columns="date", values="temp_med").columns
    
    forecast_to_hist_map = {}
    for dt in forecast_dates:
        target = dt - pd.DateOffset(years=5)
        nearest = hist_cols[np.argmin(np.abs(hist_cols - target))]
        forecast_to_hist_map[dt] = nearest
        
    mapped_cols = [forecast_to_hist_map[dt] for dt in forecast_dates]
    mapped_weeks = [min(53, max(1, get_epiweek_from_sunday(dt)[1])) for dt in mapped_cols]
    
    # Ensure forecast dates exist in clim_pred columns
    clim_pred_dates = sorted(clim_pred["date"].unique())
    forecast_dates_str = [dt.strftime("%Y-%m-%d") for dt in forecast_dates]
    
    forecast_climate = {}
    for var in ["temp_med", "precip_tot"]:
        # 1. Smooth predicted climate from models
        pred_pivot = (
            clim_pred.pivot(index="geocode", columns="date", values=var)
            .ffill(axis=1).bfill(axis=1).reindex(geocodes)
        )
        pred_matrix = pred_pivot[forecast_dates_str].values.astype(np.float32)
        
        # 2. Historical actual weather
        hist_pivot = (
            hist_df.pivot(index="geocode", columns="date", values=var)
            .ffill(axis=1).bfill(axis=1).reindex(geocodes)
        )
        hist_actual = hist_pivot[mapped_cols].values.astype(np.float32)
        
        # 3. Historical baseline
        hist_base = hist_temp_matrix if var == "temp_med" else hist_precip_matrix
        hist_baseline_mapped = hist_base[:, [w - 1 for w in mapped_weeks]]
        
        # 4. Combine: smooth predictions + historical anomalies (actual - baseline)
        hist_anomaly = hist_actual - hist_baseline_mapped
        forecast_climate[var] = pred_matrix + hist_anomaly

    m_pops = pop_series.reindex(geocodes).values.astype(np.float32)
    clim_matrix = {}
    for var in ["temp_med", "precip_tot"]:
        clim_matrix[var] = np.concatenate([hist_climate[var], forecast_climate[var]], axis=1)

    # Full history matrices
    history_inc = np.zeros((num_muns, 104 + num_weeks), dtype=np.float32)
    history_inc[:, :104] = hist_inc_pivot
    history_cases = np.zeros((num_muns, 104 + num_weeks), dtype=np.float32)
    history_cases[:, :104] = hist_cases_pivot

    # Zone specific difference scales and gammas
    # For recursive multi-step forecasting, we use larger scales for the expanding zones (4, 5, 6)
    # to allow the compounded feedback loop to reach realistic outbreak magnitudes (matching 2022-2024).
    zone_diff_scales = {
        1.0: 0.45, 2.0: 0.45, 3.0: 0.45,
        4.0: 1.0,  5.0: 1.2,  6.0: 1.2
    }
    
    zone_gammas = {
        1.0: 0.02, 2.0: 0.02, 3.0: 0.02,
        4.0: 0.02, 5.0: 0.02, 6.0: 0.02
    }

    # Annual growth rates for baseline (moderated to prevent over-inflation)
    zone_growth = {
        1.0: 0.0,  2.0: 0.0,  3.0: 0.0,
        4.0: 0.02, 5.0: 0.02, 6.0: 0.03
    }
    
    diff_scales = np.array([zone_diff_scales.get(z, 0.45) for z in climate_zones], dtype=np.float32)
    gammas = np.array([zone_gammas.get(z, 0.08) for z in climate_zones], dtype=np.float32)
    growth_rates = np.array([zone_growth.get(z, 0.0) for z in climate_zones], dtype=np.float32)

    X_t = np.zeros((num_muns, 12), dtype=np.float32)

    print("\nRunning recursive dengue forecasting loop with stochastic noise...")

    for t in range(num_weeks):
        wk = week_seq[t]
        
        # Apply annual baseline growth to represent geographic expansion
        years_passed = t / 52.0
        growth_multiplier = (1.0 + growth_rates) ** years_passed
        baseline = hist_mean_matrix[:, wk - 1] * growth_multiplier

        # Build feature vector
        X_t[:, 0] = baseline
        X_t[:, 1] = m_pops
        X_t[:, 2] = history_inc[:, 104 + t - 1]
        X_t[:, 3] = history_inc[:, 104 + t - 1] - history_inc[:, 104 + t - 2]
        X_t[:, 4] = history_inc[:, 104 + t - 2]
        X_t[:, 5] = history_inc[:, 104 + t - 52]
        X_t[:, 6] = history_inc[:, 104 + t - 104]
        cum_cases = history_cases[:, 104 + t - 52: 104 + t].sum(axis=1)
        X_t[:, 7] = (cum_cases / np.maximum(m_pops, 1)) * 100000.0
        temp_val = clim_matrix["temp_med"][:, 8 + t]
        X_t[:, 8] = temp_val
        temp_anom = temp_val - hist_temp_matrix[:, wk - 1]
        X_t[:, 9] = temp_anom
        precip_val = clim_matrix["precip_tot"][:, 8 + t]
        X_t[:, 10] = precip_val
        X_t[:, 11] = precip_val - hist_precip_matrix[:, wk - 1]

        X_t_clean = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)

        # Zone-wise model prediction
        preds_diff = np.zeros(num_muns, dtype=np.float32)
        for zone in range(1, 7):
            zone_mask = climate_zones == float(zone)
            if np.any(zone_mask):
                preds_diff[zone_mask] = models[float(zone)].predict(X_t_clean[zone_mask])

        # Mean-reversion correction with zone-specific scales and gammas
        dev = history_inc[:, 104 + t - 1] - baseline
        
        # Apply climate suitability multiplier to change term (beta = 0.20)
        suit = np.clip(1.0 + 0.20 * temp_anom, 0.5, 2.5)
        
        # Add temperature-driven spark to ignite outbreaks in warm springs
        # If temp_anom is positive and current incidence is low, add a small spark
        spark = np.where((temp_anom > 0.0) & (history_inc[:, 104 + t - 1] < 15.0), 0.35 * temp_anom, 0.0)
        
        preds = history_inc[:, 104 + t - 1] + diff_scales * (preds_diff * suit + spark) - gammas * dev

        # ----------------------------------------------------------------
        # Inject local municipality noise + small correlated zone noise
        # ----------------------------------------------------------------
        noise_std = noise_std_by_week[wk - 1]
        
        # Local municipality-level noise (20% scale)
        local_noise = rng.normal(loc=0.0, scale=0.20 * noise_std, size=num_muns).astype(np.float32)
        
        # Small zone-level noise (adds realistic wiggles to the zone averages, scale=0.8 cases per 100k)
        unique_zones = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        zone_noise_vals = rng.normal(loc=0.0, scale=0.8, size=len(unique_zones)).astype(np.float32)
        zone_noise_map = {z: val for z, val in zip(unique_zones, zone_noise_vals)}
        zone_noise = np.array([zone_noise_map.get(z, 0.0) for z in climate_zones], dtype=np.float32)
        
        preds = preds + local_noise + zone_noise

        # Clip to a minimum of 1% of baseline to prevent zero flatlining
        preds = np.clip(preds, 0.01 * baseline, None)

        history_inc[:, 104 + t] = preds
        history_cases[:, 104 + t] = np.round(preds * m_pops / 100000.0)

    dengue_inc_forecast = history_inc[:, 104:]
    dengue_cases_forecast = history_cases[:, 104:]

    print("\nAssembling forecast dataframe...")
    week_dfs = []
    for t in range(num_weeks):
        wk_df = pd.DataFrame(index=geocodes)
        wk_df["date"] = forecast_dates[t].strftime("%Y-%m-%d")
        wk_df["year"] = forecast_years[t]
        wk_df["week"] = forecast_weeks[t]
        wk_df["geocode"] = geocodes
        wk_df["uf"] = meta_df["uf"].values
        wk_df["climate_zone"] = meta_df["climate_zone"].values
        wk_df["population"] = pop_series.reindex(geocodes).values
        wk_df["incidence_rate"] = dengue_inc_forecast[:, t]
        wk_df["cases"] = dengue_cases_forecast[:, t]
        week_dfs.append(wk_df)

    forecast_df = pd.concat(week_dfs, ignore_index=True)
    # Keep the entire forecasted range (from June 2024 to December 2029)
    # to avoid a 6-month gap between history and forecast.
    forecast_2025_2029 = forecast_df.reset_index(drop=True)

    m_out1 = "dengue/outputs/municipality_dengue_2025_2029.csv"
    m_out2 = "final/outputs/csv/municipality_dengue_2025_2029.csv"
    forecast_2025_2029.to_csv(m_out1, index=False)
    forecast_2025_2029.to_csv(m_out2, index=False)
    print(f"Saved municipality dengue forecasts to {m_out1} and {m_out2}")

    # Zone aggregation using correct zone-level populations
    print("\nAggregating predictions into Climate Zones...")
    def aggregate_zones(group):
        date_val = group["date"].iloc[0]
        year_val = group["year"].iloc[0]
        week_val = group["week"].iloc[0]
        records = []
        for zone in sorted(group["climate_zone"].unique()):
            zg = group[group["climate_zone"] == zone]
            zone_cases = zg["cases"].sum()
            # Use correct zone population = sum of all municipality populations in zone
            zone_pop = zone_pop_dict.get(zone, zg["population"].sum())
            zone_incidence = (zone_cases / zone_pop) * 100000.0 if zone_pop > 0 else 0.0
            records.append({
                "date": date_val, "year": year_val, "week": week_val,
                "climate_zone": zone, "population": zone_pop,
                "cases": zone_cases, "incidence_rate": zone_incidence
            })
        return pd.DataFrame(records)

    zone_df = forecast_2025_2029.groupby("date", group_keys=False).apply(aggregate_zones).reset_index(drop=True)

    z_out1 = "dengue/outputs/zone_dengue_2025_2029.csv"
    z_out2 = "final/outputs/csv/zone_dengue_2025_2029.csv"
    zone_df.to_csv(z_out1, index=False)
    zone_df.to_csv(z_out2, index=False)
    print(f"Saved zone-level dengue forecasts to {z_out1} and {z_out2}")

if __name__ == "__main__":
    predict_dengue()
