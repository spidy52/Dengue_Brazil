import os
import gc
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore")

# Set premium visualization styles
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "grid.alpha": 0.3
})

def generate_visualizations_2years():
    print("Generating 2-Year Visualizations...")
    
    # Paths
    hist_csv = "final_brazil_dengue.csv"
    if not os.path.exists(hist_csv):
        hist_csv = os.path.join("data", "final_brazil_dengue.csv")
    climate_pred_csv = "final/outputs/csv/municipality_climate_2025_2029.csv"
    dengue_pred_csv = "final/outputs_2years/csv/municipality_dengue_2025_2026.csv"
    zone_pred_csv = "final/outputs_2years/csv/zone_dengue_2025_2026.csv"
    
    # Create graphs directory
    os.makedirs("final/outputs_2years/graphs", exist_ok=True)
    
    # 1. Load History (last few years to keep memory low, but enough for graphs)
    print("Loading historical data for context...")
    hist_df = pd.read_csv(hist_csv, usecols=["date", "year", "epiweek", "geocode", "uf", "climate_zone", "population", "cases", "incidence_rate", "temp_med", "precip_tot"])
    hist_df = hist_df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    
    # Load Forecasts
    print("Loading forecasted data...")
    clim_fore = pd.read_csv(climate_pred_csv)
    clim_fore["date"] = pd.to_datetime(clim_fore["date"])
    clim_fore = clim_fore[clim_fore["date"] <= "2026-12-31"] # Filter for 2 years
    
    deng_fore = pd.read_csv(dengue_pred_csv)
    deng_fore["date"] = pd.to_datetime(deng_fore["date"])
    
    zone_fore = pd.read_csv(zone_pred_csv)
    zone_fore["date"] = pd.to_datetime(zone_fore["date"])
    
    # ----------------------------------------------------
    # PART 1: DENGUE VALIDATION TIME-SERIES CURVES
    # ----------------------------------------------------
    print("Generating Dengue Validation Curves per Zone...")
    models = {}
    for zone in range(1, 7):
        m_path = f"dengue/models/dengue_model_zone_{zone}.joblib"
        if os.path.exists(m_path):
            models[float(zone)] = joblib.load(m_path)
            
    if len(models) == 6:
        val_df = hist_df.copy()
        for col in val_df.columns:
            if val_df[col].dtype == np.float64:
                val_df[col] = val_df[col].astype(np.float32)
            elif val_df[col].dtype == np.int64:
                val_df[col] = val_df[col].astype(np.int32)
        val_df["week"] = (val_df["epiweek"] % 100).astype(np.int16)
        
        val_df["lag_1_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(1).astype(np.float32)
        val_df["lag_2_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(2).astype(np.float32)
        val_df["lag_52_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(52).astype(np.float32)
        val_df["lag_104_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(104).astype(np.float32)
        val_df["diff_lag_1"] = val_df["lag_1_incidence"] - val_df["lag_2_incidence"]
        
        val_df["rolling_cases_52w"] = val_df.groupby("geocode")["cases"].shift(1).rolling(window=52, min_periods=1).sum().reset_index(level=0, drop=True).astype(np.float32)
        val_df["cum_incidence_52w"] = (val_df["rolling_cases_52w"] / val_df["population"]) * 100000.0
        
        train_mask = val_df["year"] <= 2022
        
        hist_mean_inc = val_df[train_mask].groupby(["geocode", "week"])["incidence_rate"].mean().reset_index().rename(columns={"incidence_rate": "hist_mean_incidence"})
        val_df = val_df.merge(hist_mean_inc, on=["geocode", "week"], how="left")
        overall_week_mean = val_df[train_mask].groupby("week")["incidence_rate"].mean().to_dict()
        val_df["hist_mean_incidence"] = val_df["hist_mean_incidence"].fillna(val_df["week"].map(overall_week_mean))
        val_df["hist_mean_incidence"] = val_df["hist_mean_incidence"].fillna(val_df["incidence_rate"].mean()).astype(np.float32)
        
        hist_temp = val_df[train_mask].groupby(["geocode", "week"])["temp_med"].mean().reset_index().rename(columns={"temp_med": "hist_mean_temp"})
        val_df = val_df.merge(hist_temp, on=["geocode", "week"], how="left")
        val_df["hist_mean_temp"] = val_df["hist_mean_temp"].fillna(val_df["temp_med"].mean()).astype(np.float32)
        val_df["temp_anomaly"] = val_df["temp_med"] - val_df["hist_mean_temp"]
        
        hist_precip = val_df[train_mask].groupby(["geocode", "week"])["precip_tot"].mean().reset_index().rename(columns={"precip_tot": "hist_mean_precip"})
        val_df = val_df.merge(hist_precip, on=["geocode", "week"], how="left")
        val_df["hist_mean_precip"] = val_df["hist_mean_precip"].fillna(val_df["precip_tot"].mean()).astype(np.float32)
        val_df["precip_anomaly"] = val_df["precip_tot"] - val_df["hist_mean_precip"]
        
        features = [
            "hist_mean_incidence", "population", "lag_1_incidence", "diff_lag_1",
            "lag_2_incidence", "lag_52_incidence", "lag_104_incidence", "cum_incidence_52w",
            "temp_med", "temp_anomaly", "precip_tot", "precip_anomaly"
        ]
        
        val_slice = val_df[(val_df["year"] >= 2022) & (val_df["year"] <= 2024)].dropna(subset=features)
        
        if len(val_slice) > 0:
            preds_diff = np.zeros(len(val_slice), dtype=np.float32)
            for zone in range(1, 7):
                zone_mask = val_slice["climate_zone"] == float(zone)
                if np.any(zone_mask):
                    preds_diff[zone_mask] = models[float(zone)].predict(val_slice[features][zone_mask])
            preds_inc = val_slice["lag_1_incidence"].values + 0.45 * preds_diff
            preds_inc = np.clip(preds_inc, 0, None)
            val_slice["predicted_incidence"] = preds_inc
            
            # Save validation time series graphs to outputs_2years/graphs/
            zone_pop_dict = val_slice.groupby("geocode")["population"].first().to_dict()
            
            for zone in range(1, 7):
                zs = val_slice[val_slice["climate_zone"] == float(zone)]
                if len(zs) == 0:
                    continue
                zg_grouped = zs.groupby("date").apply(
                    lambda g: pd.Series({
                        "actual_incidence": (g["cases"].sum() / sum(zone_pop_dict.get(m, 0) for m in g["geocode"].unique()) * 100000.0) if len(g) > 0 else 0.0,
                        "predicted_incidence": ( (g["predicted_incidence"] * g["population"]).sum() / g["population"].sum() ) if len(g) > 0 else 0.0
                    })
                ).reset_index()
                
                zg_grouped["predicted_incidence_smooth"] = zg_grouped["predicted_incidence"].rolling(3, center=True, min_periods=1).mean()
                
                plt.figure(figsize=(14, 6))
                plt.plot(zg_grouped["date"], zg_grouped["actual_incidence"], label="Actual", color="#1f77b4", lw=2)
                plt.plot(zg_grouped["date"], zg_grouped["predicted_incidence_smooth"], label="Predicted", color="#ff7f0e", lw=2)
                
                plt.title(f"LightGBM Validation - Climate Zone {int(zone)} (Actual vs Predicted)", fontsize=14, fontweight="bold", pad=15)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Incidence Rate (per 100k)", fontsize=12)
                plt.legend(loc="upper right", fontsize=10)
                plt.tight_layout()
                plt.savefig(f"final/outputs_2years/graphs/dengue_val_time_series_zone_{int(zone)}.png", dpi=150)
                plt.close()
                
        del val_df, val_slice, models
        gc.collect()

    # ----------------------------------------------------
    # PART 2: DENGUE FORECAST VS HISTORY COMPARISON
    # ----------------------------------------------------
    print("Generating Dengue Forecast plots with history...")
    hist_recent = hist_df[hist_df["year"] >= 2020]
    
    zone_pop_dict = (
        hist_df[hist_df["date"].dt.year == 2024]
        .groupby(["geocode","climate_zone"])["population"].first()
        .reset_index()
        .groupby("climate_zone")["population"].sum()
        .to_dict()
    )
    
    def agg_hist_zones(group):
        date_val = group["date"].iloc[0]
        records = []
        for zone in sorted(group["climate_zone"].unique()):
            zg = group[group["climate_zone"] == zone]
            z_cases = zg["cases"].sum()
            z_pop = zone_pop_dict.get(zone, zg["population"].sum())
            z_inc = (z_cases / z_pop * 100000.0) if z_pop > 0 else 0.0
            records.append({
                "date": date_val, "climate_zone": zone, "cases": z_cases, "incidence_rate": z_inc
            })
        return pd.DataFrame(records)
        
    hist_zone_df = hist_recent.groupby("date", group_keys=False).apply(agg_hist_zones).reset_index(drop=True)
    
    # Plot history + forecast for each zone
    for zone in sorted(zone_fore["climate_zone"].unique()):
        hz = hist_zone_df[hist_zone_df["climate_zone"] == zone].sort_values("date")
        fz = zone_fore[zone_fore["climate_zone"] == zone].sort_values("date")
        
        hist_peak = hz["incidence_rate"].max()
        fore_peak = fz["incidence_rate"].max()
        
        fore_std = fz["incidence_rate"].rolling(4, center=True, min_periods=1).std().fillna(0)
        fore_upper = fz["incidence_rate"] + fore_std
        fore_lower = (fz["incidence_rate"] - fore_std).clip(lower=0)
        
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        use_dual = (hist_peak > 0) and (fore_peak > 0) and (max(hist_peak / fore_peak, fore_peak / hist_peak) > 3.0)
        
        if use_dual:
            ax2 = ax1.twinx()
            ax1.plot(hz["date"], hz["incidence_rate"], label="Historical", color="#1f77b4", lw=2.5)
            ax2.plot(fz["date"], fz["incidence_rate"], label="Forecast (right axis)", color="#ff7f0e", lw=2.5, linestyle="--")
            ax2.fill_between(fz["date"], fore_lower, fore_upper, color="#ff7f0e", alpha=0.18, label="Forecast ±1σ")
            ax1.set_ylabel("Historical Incidence Rate (per 100k)", color="#1f77b4")
            ax2.set_ylabel("Forecast Incidence Rate (per 100k)", color="#ff7f0e")
            ax1.tick_params(axis="y", labelcolor="#1f77b4")
            ax2.tick_params(axis="y", labelcolor="#ff7f0e")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
            ax1.set_title(
                f"Dengue Incidence Rate - Climate Zone {zone} - History vs Forecast\n"
                f"(Note: dual Y-axis — historical peak {hist_peak:.0f}, forecast peak {fore_peak:.0f} per 100k)",
                fontweight="bold"
            )
        else:
            ax1.plot(hz["date"], hz["incidence_rate"], label="Historical", color="#1f77b4", lw=2.5)
            ax1.plot(fz["date"], fz["incidence_rate"], label="Forecast", color="#ff7f0e", lw=2.5, linestyle="--")
            ax1.fill_between(fz["date"], fore_lower, fore_upper, color="#ff7f0e", alpha=0.18, label="Forecast ±1σ")
            ax1.set_ylabel("Incidence Rate (per 100k)")
            ax1.legend()
            ax1.set_title(f"Dengue Incidence Rate - Climate Zone {zone} - History vs Forecast", fontweight="bold")
            
        forecast_start = fz["date"].min()
        ax1.axvline(x=forecast_start, color="gray", linestyle=":", lw=1.5, alpha=0.7)
        ax1.text(forecast_start, ax1.get_ylim()[1] * 0.95, "  Forecast →", color="gray", va="top")
        ax1.set_xlabel("Date")
        plt.tight_layout()
        plt.savefig(f"final/outputs_2years/graphs/dengue_forecast_zone_{zone}.png", dpi=150)
        plt.close()
        
    # Combined zone forecast line plot
    plt.figure(figsize=(14, 7))
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    for i, zone in enumerate(sorted(zone_fore["climate_zone"].unique())):
        fz = zone_fore[zone_fore["climate_zone"] == zone].sort_values("date")
        c = colors[i % len(colors)]
        plt.plot(fz["date"], fz["incidence_rate"], label=f"Zone {zone}", lw=2, color=c)
        fstd = fz["incidence_rate"].rolling(4, center=True, min_periods=1).std().fillna(0)
        plt.fill_between(fz["date"], (fz["incidence_rate"]-fstd).clip(0), fz["incidence_rate"]+fstd, alpha=0.10, color=c)
    plt.title("Dengue Incidence Forecast by Climate Zone (2025–2026)", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Incidence Rate (per 100k)", fontsize=12)
    plt.legend(title="Climate Zone", fontsize=10)
    plt.tight_layout()
    plt.savefig("final/outputs_2years/graphs/dengue_forecast_combined_zones.png", dpi=150)
    plt.close()
    
    print("Visualizations generated successfully. All plots saved to final/outputs_2years/graphs/")

if __name__ == "__main__":
    generate_visualizations_2years()
