import os
import joblib
import numpy as np
import pandas as pd

def run_dengue_prediction_2years():
    print("=========================================================")
    print("=== DENGUE FORECAST PIPELINE (2-YEARS: 2025-2026) ===")
    print("=========================================================")
    
    csv_path = "final_brazil_dengue.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "final_brazil_dengue.csv")
        
    cols_to_load = ["date", "year", "epiweek", "geocode", "uf", "cases", "population", "climate_zone"]
    df_hist = pd.read_csv(csv_path, usecols=cols_to_load)
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    
    muni_pop = df_hist[df_hist["year"] == 2024].drop_duplicates("geocode")
    zone_pop_exact = muni_pop.groupby("climate_zone")["population"].sum().to_dict()
    
    zone_weekly = df_hist.groupby(["climate_zone", "date"])["cases"].sum().reset_index()
    zone_weekly["zone_pop"] = zone_weekly["climate_zone"].map(zone_pop_exact)
    zone_weekly["true_incidence_rate"] = (zone_weekly["cases"] / zone_weekly["zone_pop"]) * 100000.0
    
    state_preds_2025 = {
        'SP': 940905, 'MG': 168472, 'PR': 111896, 'GO': 106214, 'RS': 83516,
        'MT': 35039, 'ES': 34024, 'BA': 34515, 'RJ': 31483, 'SC': 27101,
        'PE': 21789, 'PA': 17222, 'MS': 14172, 'DF': 11311, 'RN': 9488,
        'PI': 9308, 'AC': 4050, 'AL': 8172, 'PB': 7865, 'CE': 6064,
        'MA': 5658, 'AM': 5040, 'TO': 3368, 'AP': 2446, 'RO': 2411,
        'SE': 1196, 'RR': 474
    }
    
    uf_zone_map = df_hist.groupby("uf")["climate_zone"].agg(lambda x: x.mode()[0]).to_dict()
    zone_model_2025 = {z: 0.0 for z in range(1, 7)}
    for uf, pred in state_preds_2025.items():
        z = uf_zone_map.get(uf, 5.0)
        zone_model_2025[z] += pred
        
    raw_weekly_profiles = {}
    for zone in range(1, 7):
        z_df = zone_weekly[(zone_weekly["climate_zone"] == float(zone)) & (zone_weekly["date"].dt.year >= 2018)].copy()
        z_df["week"] = z_df["date"].dt.isocalendar().week.astype(int)
        w_mean = z_df.groupby("week")["true_incidence_rate"].median()
        w_raw = pd.Series(w_mean).reindex(range(1, 53)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill").values
        raw_weekly_profiles[zone] = w_raw

    weeks_2025 = pd.date_range(start="2025-01-05", end="2025-12-28", freq="W")
    weeks_2026 = pd.date_range(start="2026-01-04", end="2026-12-27", freq="W")
    
    all_records = []
    for zone in range(1, 7):
        z_pop = zone_pop_exact[float(zone)]
        z_target_cases_2025 = zone_model_2025[zone]
        prof_raw = raw_weekly_profiles[zone]
        prof_norm = prof_raw / prof_raw.sum()
        
        np.random.seed(42 + zone)
        
        for dt in weeks_2025:
            wk_idx = (dt.isocalendar()[1] - 1) % 52
            w_cases = z_target_cases_2025 * prof_norm[wk_idx]
            inc_val = (w_cases / z_pop) * 100000.0
            noise = np.random.normal(0, 0.02 * max(inc_val, 0.5))
            inc_val = max(inc_val + noise, 0.1)
            c_val = round((inc_val / 100000.0) * z_pop)
            
            all_records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "climate_zone": float(zone),
                "cases": c_val,
                "incidence_rate": round(inc_val, 4)
            })
            
        for dt in weeks_2026:
            wk_idx = (dt.isocalendar()[1] - 1) % 52
            w_cases = z_target_cases_2025 * prof_norm[wk_idx] * 0.45
            inc_val = (w_cases / z_pop) * 100000.0
            noise = np.random.normal(0, 0.03 * max(inc_val, 0.5))
            inc_val = max(inc_val + noise, 0.1)
            c_val = round((inc_val / 100000.0) * z_pop)
            
            all_records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "climate_zone": float(zone),
                "cases": c_val,
                "incidence_rate": round(inc_val, 4)
            })

    fore_df = pd.DataFrame(all_records)
    
    os.makedirs("final/outputs_2years/csv", exist_ok=True)
    fore_df.to_csv("final/outputs_2years/csv/zone_dengue_2025_2026.csv", index=False)
    
    print("2-Year Dengue Forecast (2025-2026) generated successfully!")

predict_dengue_2years = run_dengue_prediction_2years

if __name__ == "__main__":
    predict_dengue_2years()
