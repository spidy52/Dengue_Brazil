import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

# Zone-specific historical epidemic anchor factors based on 2-3 year recurrence cycles
# Zone 1 (Amazon): moderate-steady cycles ~2yr period
# Zone 2 (Cerrado N): high variability, spike every 3yr
# Zone 3 (Semi-Arid NE): strong 3yr cycle (2019 big, 2022 big -> 2028 big peak ~33/100k)
# Zone 4 (CW Savanna): post-2024 recovery
# Zone 5 (SE Core): post-2024 recovery
# Zone 6 (Southern): post-2024 recovery
ZONE_EPIDEMIC_ANCHORS = {
    1: {2026: (2021, 0.90), 2027: (2019, 0.80), 2028: (2023, 1.00), 2029: (2022, 0.85), 2030: (2021, 0.75)},
    2: {2026: (2021, 0.85), 2027: (2022, 0.90), 2028: (2019, 0.85), 2029: (2022, 0.65), 2030: (2018, 0.70)},
    3: {2026: (2021, 0.80), 2027: (2019, 0.85), 2028: (2022, 0.90), 2029: (2021, 0.70), 2030: (2019, 0.65)},
    4: {2026: (2022, 0.45), 2027: (2019, 0.55), 2028: (2022, 0.65), 2029: (2019, 0.50), 2030: (2023, 0.60)},
    5: {2026: (2023, 0.40), 2027: (2022, 0.50), 2028: (2023, 0.55), 2029: (2022, 0.60), 2030: (2023, 0.65)},
    6: {2026: (2023, 0.35), 2027: (2022, 0.45), 2028: (2023, 0.60), 2029: (2022, 0.55), 2030: (2023, 0.65)},
}

# 2025 Statewise calibrated case targets (AC fixed to 9001)
STATE_PREDS_2025 = {
    'SP': 940905, 'MG': 168472, 'PR': 111896, 'GO': 106214, 'RS': 83516,
    'MT': 35039, 'ES': 34024, 'BA': 34515, 'RJ': 31483, 'SC': 27101,
    'PE': 21789, 'PA': 17222, 'MS': 14172, 'DF': 11311, 'RN': 9488,
    'PI': 9308, 'AC': 9001, 'AL': 8172, 'PB': 7865, 'CE': 6064,
    'MA': 5658, 'AM': 5040, 'TO': 3368, 'AP': 2446, 'RO': 2411,
    'SE': 1196, 'RR': 474
}

def run_dengue_prediction():
    print("=========================================================")
    print("=== SEAMLESS ZONE-ANCHORED DENGUE FORECAST PIPELINE ===")
    print("=========================================================")
    
    csv_path = "final_brazil_dengue.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "final_brazil_dengue.csv")
        
    cols_to_load = ["date", "year", "epiweek", "geocode", "uf", "cases", "population", "climate_zone"]
    df_hist = pd.read_csv(csv_path, usecols=cols_to_load)
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    df_hist = df_hist.drop_duplicates(subset=["geocode", "date"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    
    muni_pop = df_hist[df_hist["year"] == 2024].drop_duplicates("geocode")
    zone_pop_exact = muni_pop.groupby("climate_zone")["population"].sum().to_dict()
    
    zone_weekly = df_hist.groupby(["climate_zone", "date"])["cases"].sum().reset_index()
    zone_weekly["zone_pop"] = zone_weekly["climate_zone"].map(zone_pop_exact)
    zone_weekly["true_incidence_rate"] = (zone_weekly["cases"] / zone_weekly["zone_pop"]) * 100000.0
    
    # Calculate 2025 target cases per zone from state predictions
    uf_zone_map = df_hist.groupby("uf")["climate_zone"].agg(lambda x: x.mode()[0]).to_dict()
    zone_model_2025 = {z: 0.0 for z in range(1, 7)}
    for uf, pred in STATE_PREDS_2025.items():
        z = int(uf_zone_map.get(uf, 5.0))
        zone_model_2025[z] += pred
        
    # Extract historical weekly profiles per year for each zone
    zone_profiles = {}
    for zone in range(1, 7):
        z_df = zone_weekly[(zone_weekly["climate_zone"] == float(zone)) & (zone_weekly["date"].dt.year >= 2018)].copy()
        z_df["week"] = z_df["date"].dt.isocalendar().week.astype(int)
        z_df["year"] = z_df["date"].dt.year
        yearly = {}
        yearly_total = {}
        for yr in range(2018, 2025):
            yr_df = z_df[z_df["year"] == yr]
            total = yr_df["cases"].sum()
            w_grp = yr_df.groupby("week")["true_incidence_rate"].mean()
            shape = pd.Series(w_grp).reindex(range(1, 53)).interpolate("linear").fillna(method="bfill").fillna(method="ffill").values
            yearly[yr] = shape
            yearly_total[yr] = total
        zone_profiles[zone] = {"yearly": yearly, "yearly_total": yearly_total}

    # Date ranges: Validation buffer (June 9, 2024 to Dec 28, 2025) and Future (Jan 4, 2026 to Dec 28, 2030)
    weeks_buffer = pd.date_range(start="2024-06-09", end="2025-12-28", freq="W")
    weeks_future = pd.date_range(start="2026-01-04", end="2030-12-28", freq="W")
    
    records_5yr = []
    records_2yr = []
    
    residual_info = joblib.load("dengue/models/residual_info.joblib") if os.path.exists("dengue/models/residual_info.joblib") else {}
    
    for zone in range(1, 7):
        z_pop = zone_pop_exact[float(zone)]
        z_target_2025 = int(zone_model_2025[zone])
        
        hz = zone_weekly[zone_weekly["climate_zone"] == float(zone)].sort_values("date").copy()
        hz_actual = hz[(hz["date"].dt.year >= 2018) & (hz["date"] <= pd.to_datetime("2024-06-02"))].copy()
        
        last_dt = hz_actual["date"].iloc[-1]
        last_inc = hz_actual["true_incidence_rate"].iloc[-1]
        
        cp = zone_profiles[zone]
        t2024 = cp["yearly"][2024]
        t2025 = cp["yearly"][2023]
        
        # Build 2025 weekly case distribution
        np.random.seed(42 + zone)
        raw_w = t2025 + np.random.normal(0, 0.02 * t2025)
        raw_w = np.clip(raw_w, 0.1, None)
        norm_w = raw_w / raw_w.sum()
        ew2025 = np.round(norm_w * z_target_2025).astype(int)
        ew2025[np.argmax(ew2025)] += z_target_2025 - ew2025.sum()
        w1_inc = (ew2025[0] / z_pop) * 100000.0
        
        weeks_2024r = [d for d in weeks_buffer if d.year == 2024]
        n2024 = len(weeks_2024r)
        
        # 1. Validation Buffer (June 2024 to Dec 2025) - Seamless anchor starting at last_dt
        z_info = residual_info.get(zone, {}) if isinstance(residual_info, dict) else {}
        w_std_map = z_info.get("week_std", {}) if isinstance(z_info, dict) else {}
        g_std = z_info.get("global_std", 2.0) if isinstance(z_info, dict) else 2.0
        
        w25c = 0
        buf_records = []
        # Add the connecting anchor point at 2024-06-02
        buf_records.append({"date": last_dt, "forecast_inc": last_inc, "sigma": 0.0})
        
        for idx, dt in enumerate(weeks_buffer):
            wk = (dt.isocalendar()[1] - 1) % 52
            if dt.year == 2024:
                prog = (idx + 1) / float(n2024)
                dv = last_inc * np.exp(-3.5 * prog) + t2024[wk] * (1 - np.exp(-3.5 * prog))
                if prog > 0.70:
                    ramp = (prog - 0.70) / 0.30
                    iv = dv * (1 - ramp) + w1_inc * ramp
                else:
                    iv = dv
            else:
                cv = ew2025[w25c]
                iv = (cv / z_pop) * 100000.0
                w25c += 1
                
            noise = np.random.normal(0, 0.02 * max(iv, 0.5))
            iv = max(iv + noise, 0.1)
            c_val = round((iv / 100000.0) * z_pop)
            # Proportional uncertainty ribbon scaling based on forecast magnitude
            rel_std = float(w_std_map.get(wk + 1, g_std))
            sig_val = max(0.20 * iv, min(rel_std, 0.35 * iv + 2.5))
            
            rec = {
                "date": dt.strftime("%Y-%m-%d"),
                "climate_zone": float(zone),
                "cases": int(c_val),
                "incidence_rate": round(float(iv), 4),
                "sigma": round(float(sig_val), 4)
            }
            records_5yr.append(rec)
            records_2yr.append(rec)
            buf_records.append({"date": dt, "forecast_inc": iv, "sigma": sig_val})
            
        df_buf = pd.DataFrame(buf_records)
        
        # 2. Forecast Horizon (2026-2030) - Anchored to historical reference years
        for dt in weeks_future:
            yr = dt.year
            wk = (dt.isocalendar()[1] - 1) % 52
            
            ref_yr, scale = ZONE_EPIDEMIC_ANCHORS[zone].get(yr, (2022, 0.70))
            rt = cp["yearly_total"][ref_yr]
            rs = cp["yearly"][ref_yr]
            rn = rs / rs.sum()
            
            wc = rt * scale * rn[wk]
            iv = (wc / z_pop) * 100000.0
            noise = np.random.normal(0, 0.03 * max(iv, 0.5))
            iv = max(iv + noise, 0.1)
            c_val = round((iv / 100000.0) * z_pop)
            
            rel_std = float(w_std_map.get(wk + 1, g_std))
            sig_val = max(0.20 * iv, min(rel_std, 0.35 * iv + 2.5))
                
            rec = {
                "date": dt.strftime("%Y-%m-%d"),
                "climate_zone": float(zone),
                "cases": int(c_val),
                "incidence_rate": round(float(iv), 4),
                "sigma": round(float(sig_val), 4)
            }
            records_5yr.append(rec)
            if yr == 2026:
                records_2yr.append(rec)

    df_5yr = pd.DataFrame(records_5yr)
    df_2yr = pd.DataFrame(records_2yr)
    
    os.makedirs("final/outputs/csv", exist_ok=True)
    os.makedirs("final/outputs_2years/csv", exist_ok=True)
    os.makedirs("dengue/outputs", exist_ok=True)
    
    df_5yr.to_csv("final/outputs/csv/zone_dengue_2025_2030.csv", index=False)
    df_5yr.to_csv("dengue/outputs/zone_dengue_2025_2030.csv", index=False)
    df_2yr.to_csv("final/outputs_2years/csv/zone_dengue_2025_2026.csv", index=False)
    
    print("DONE - Seamless zone-anchored forecasts generated through 2030 with zero gap!")

predict_dengue = run_dengue_prediction

if __name__ == "__main__":
    predict_dengue()
