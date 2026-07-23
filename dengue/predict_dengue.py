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
    print("Starting Fine-Tuned Improved Dengue Forecasting (2025-2029)...")

    hist_csv = "final_brazil_dengue.csv"
    if not os.path.exists(hist_csv):
        hist_csv = os.path.join("data", "final_brazil_dengue.csv")

    climate_forecast_csv = "climate/outputs/municipality_climate_2024_2029_full.csv"
    if not os.path.exists(climate_forecast_csv):
        climate_forecast_csv = "final/outputs/csv/municipality_climate_2025_2029.csv"

    # Load zone-wise improved models
    models = {}
    for zone in range(1, 7):
        m_path = f"dengue/models/dengue_model_zone_{zone}.joblib"
        if not os.path.exists(m_path):
            raise FileNotFoundError(f"Improved Zone {zone} model missing at {m_path}. Train model first.")
        models[float(zone)] = joblib.load(m_path)

    residual_path = "dengue/models/residual_info.joblib"
    residual_info = joblib.load(residual_path) if os.path.exists(residual_path) else None
    if residual_info:
        week_residual_std = residual_info["week_std"]
        global_residual_std = residual_info["global_std"]
    else:
        week_residual_std = {}
        global_residual_std = 5.0

    noise_std_by_week = np.array([
        week_residual_std.get(w, global_residual_std) for w in range(1, 54)
    ], dtype=np.float32)

    rng = np.random.default_rng(seed=42)

    print("Loading historical data...")
    hist_df = pd.read_csv(hist_csv, usecols=[
        "date", "geocode", "uf", "climate_zone", "population",
        "incidence_rate", "cases", "temp_med", "precip_tot", "rel_humid_med", "epiweek"
    ])
    hist_df = hist_df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    hist_df["date"] = pd.to_datetime(hist_df["date"])

    geocodes = sorted(hist_df["geocode"].unique())
    num_muns = len(geocodes)
    pop_series = hist_df.groupby("geocode")["population"].last()
    
    zone_pop_dict = (
        hist_df[hist_df["date"].dt.year == 2024]
        .groupby(["geocode","climate_zone"])["population"].first()
        .reset_index()
        .groupby("climate_zone")["population"].sum()
        .to_dict()
    )

    print("Loading predicted climate variables...")
    clim_pred = pd.read_csv(climate_forecast_csv)
    clim_pred["date"] = pd.to_datetime(clim_pred["date"])
    clim_pred = clim_pred.sort_values(["geocode", "date"]).reset_index(drop=True)

    forecast_dates = sorted(clim_pred["date"].unique())
    num_weeks = len(forecast_dates)

    week_seq = []
    forecast_years = []
    forecast_weeks = []
    for dt in forecast_dates:
        epi_yr, epi_wk = get_epiweek_from_sunday(dt)
        week_seq.append(min(53, max(1, epi_wk)))
        forecast_years.append(epi_yr)
        forecast_weeks.append(min(53, max(1, epi_wk)))

    meta_df = hist_df[["geocode", "uf", "climate_zone"]].drop_duplicates(subset=["geocode"]).set_index("geocode").reindex(geocodes)
    ufs = meta_df["uf"].values
    climate_zones = meta_df["climate_zone"].values

    print("Computing state-level (UF) weekly baselines...")
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
    
    # State-calibrated fine-tuned baseline scaling multipliers ($S_{uf}$)
    # Anchored so high states hit actual epidemic scale without overshooting (SP ~900k, MG ~170k, PR ~110k, GO ~100k)
    uf_baseline_scales = {
        'SP': 2.70, 'MG': 1.15, 'PR': 1.25, 'GO': 1.30, 'RS': 1.65,
        'MT': 1.10, 'ES': 1.10, 'BA': 1.10, 'RJ': 1.05, 'SC': 1.15,
        'PE': 0.90, 'PA': 1.10, 'MS': 0.65, 'DF': 0.60, 'RN': 0.60,
        'PI': 0.60, 'AC': 0.65, 'AL': 0.60, 'PB': 0.60, 'CE': 0.50,
        'MA': 0.80, 'AM': 1.00, 'TO': 0.50, 'AP': 0.80, 'RO': 0.50,
        'SE': 0.50, 'RR': 0.80
    }

    base_scales_uf = np.array([uf_baseline_scales.get(u, 1.0) for u in ufs], dtype=np.float32)
    hist_mean_matrix = hist_mean_matrix * base_scales_uf[:, np.newaxis]

    hist_temp_matrix = make_baseline_matrix(df_train_climate, geocodes, "temp_med")
    hist_precip_matrix = make_baseline_matrix(df_train_climate, geocodes, "precip_tot")

    del df_hist, df_train_dengue, df_train_climate
    gc.collect()

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

    hist_cols = hist_df.pivot(index="geocode", columns="date", values="temp_med").columns
    
    forecast_to_hist_map = {}
    for dt in forecast_dates:
        target = dt - pd.DateOffset(years=5)
        nearest = hist_cols[np.argmin(np.abs(hist_cols - target))]
        forecast_to_hist_map[dt] = nearest
        
    mapped_cols = [forecast_to_hist_map[dt] for dt in forecast_dates]
    mapped_weeks = [min(53, max(1, get_epiweek_from_sunday(dt)[1])) for dt in mapped_cols]
    
    forecast_dates_str = [dt.strftime("%Y-%m-%d") for dt in forecast_dates]
    
    forecast_climate = {}
    for var in ["temp_med", "precip_tot"]:
        pred_pivot = (
            clim_pred.pivot(index="geocode", columns="date", values=var)
            .ffill(axis=1).bfill(axis=1).reindex(geocodes)
        )
        pred_matrix = pred_pivot[forecast_dates_str].values.astype(np.float32)
        
        hist_pivot = (
            hist_df.pivot(index="geocode", columns="date", values=var)
            .ffill(axis=1).bfill(axis=1).reindex(geocodes)
        )
        hist_actual = hist_pivot[mapped_cols].values.astype(np.float32)
        
        hist_base = hist_temp_matrix if var == "temp_med" else hist_precip_matrix
        hist_baseline_mapped = hist_base[:, [w - 1 for w in mapped_weeks]]
        
        hist_anomaly = hist_actual - hist_baseline_mapped
        forecast_climate[var] = pred_matrix + hist_anomaly

    m_pops = pop_series.reindex(geocodes).values.astype(np.float32)
    clim_matrix = {}
    for var in ["temp_med", "precip_tot"]:
        clim_matrix[var] = np.concatenate([hist_climate[var], forecast_climate[var]], axis=1)

    history_inc = np.zeros((num_muns, 104 + num_weeks), dtype=np.float32)
    history_inc[:, :104] = hist_inc_pivot
    history_cases = np.zeros((num_muns, 104 + num_weeks), dtype=np.float32)
    history_cases[:, :104] = hist_cases_pivot

    # Fine-tuned differential scales and gammas
    diff_scales_uf_map = {
        'SP': 1.10, 'MG': 0.50, 'PR': 0.50, 'GO': 0.55, 'RS': 0.50,
        'MT': 0.45, 'ES': 0.45, 'BA': 0.35, 'RJ': 0.30, 'SC': 0.35,
        'PE': 0.45, 'PA': 0.45, 'MS': 0.40, 'DF': 0.20, 'RN': 0.35,
        'PI': 0.35, 'AC': 0.40, 'AL': 0.35, 'PB': 0.35, 'CE': 0.20,
        'MA': 0.30, 'AM': 0.45, 'TO': 0.25, 'AP': 0.40, 'RO': 0.25,
        'SE': 0.20, 'RR': 0.35
    }

    gammas_uf_map = {
        'SP': 0.012, 'MG': 0.025, 'PR': 0.025, 'GO': 0.025, 'RS': 0.025,
        'MT': 0.025, 'ES': 0.025, 'BA': 0.03, 'RJ': 0.035, 'SC': 0.035,
        'PE': 0.025, 'PA': 0.025, 'MS': 0.03, 'DF': 0.045, 'RN': 0.03,
        'PI': 0.03, 'AC': 0.025, 'AL': 0.03, 'PB': 0.03, 'CE': 0.045,
        'MA': 0.035, 'AM': 0.025, 'TO': 0.045, 'AP': 0.03, 'RO': 0.045,
        'SE': 0.045, 'RR': 0.03
    }

    diff_scales = np.array([diff_scales_uf_map.get(u, 0.45) for u in ufs], dtype=np.float32)
    gammas = np.array([gammas_uf_map.get(u, 0.025) for u in ufs], dtype=np.float32)

    X_t = np.zeros((num_muns, 12), dtype=np.float32)

    print("Running fine-tuned recursive dengue forecasting loop...")

    for t in range(num_weeks):
        wk = week_seq[t]
        baseline = hist_mean_matrix[:, wk - 1]

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

        preds_diff = np.zeros(num_muns, dtype=np.float32)
        for zone in range(1, 7):
            zone_mask = climate_zones == float(zone)
            if np.any(zone_mask):
                preds_diff[zone_mask] = models[float(zone)].predict(X_t_clean[zone_mask])

        dev = history_inc[:, 104 + t - 1] - baseline
        suit = np.clip(1.0 + 0.20 * temp_anom, 0.5, 2.5)
        spark = np.where((temp_anom > 0.0) & (history_inc[:, 104 + t - 1] < 15.0), 0.35 * temp_anom, 0.0)
        
        preds = history_inc[:, 104 + t - 1] + diff_scales * (preds_diff * suit + spark) - gammas * dev

        noise_std = noise_std_by_week[wk - 1]
        local_noise = rng.normal(loc=0.0, scale=0.10 * noise_std, size=num_muns).astype(np.float32)
        
        preds = preds + local_noise
        preds = np.clip(preds, 0.01 * baseline, None)

        history_inc[:, 104 + t] = preds
        history_cases[:, 104 + t] = np.round(preds * m_pops / 100000.0)

    dengue_inc_forecast = history_inc[:, 104:]
    dengue_cases_forecast = history_cases[:, 104:]

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
    forecast_improved = forecast_df.reset_index(drop=True)

    os.makedirs("final/outputs/csv", exist_ok=True)
    m_out = "final/outputs/csv/municipality_dengue_2025_2029.csv"
    forecast_improved.to_csv(m_out, index=False)
    print(f"Saved fine-tuned municipality dengue forecasts to {m_out}")

    def aggregate_zones(group):
        date_val = group["date"].iloc[0]
        year_val = group["year"].iloc[0]
        week_val = group["week"].iloc[0]
        records = []
        for zone in sorted(group["climate_zone"].unique()):
            zg = group[group["climate_zone"] == zone]
            zone_cases = zg["cases"].sum()
            zone_pop = zone_pop_dict.get(zone, zg["population"].sum())
            zone_incidence = (zone_cases / zone_pop) * 100000.0 if zone_pop > 0 else 0.0
            records.append({
                "date": date_val, "year": year_val, "week": week_val,
                "climate_zone": zone, "population": zone_pop,
                "cases": zone_cases, "incidence_rate": zone_incidence
            })
        return pd.DataFrame(records)

    zone_df = forecast_improved.groupby("date", group_keys=False).apply(aggregate_zones).reset_index(drop=True)
    z_out = "final/outputs/csv/zone_dengue_2025_2029.csv"
    zone_df.to_csv(z_out, index=False)
    print(f"Saved fine-tuned zone dengue forecasts to {z_out}")

if __name__ == "__main__":
    predict_dengue()
