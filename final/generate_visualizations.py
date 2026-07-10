import os
import gc
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

TOP_FEATURES = [
    "hist_mean_incidence",
    "population",
    "lag_1_incidence",
    "diff_lag_1",
    "lag_2_incidence",
    "cases_per_week_change",
    "temp_med",
    "lag_4_humid",
    "lag_2_rain",
    "precip_tot"
]

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

def generate_features_for_target(df, target):
    lags = [1, 2, 3, 4, 8, 12, 16, 20, 26, 39, 52]
    features = []
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby("geocode")[target].shift(lag).astype(np.float32)
        features.append(col_name)
    windows = [4, 8, 12, 26, 52]
    shifted = df.groupby("geocode")[target].shift(1)
    for w in windows:
        mean_col = f"roll_mean_{w}"
        df[mean_col] = shifted.rolling(w).mean().astype(np.float32)
        features.append(mean_col)
    for w in windows:
        std_col = f"roll_std_{w}"
        df[std_col] = shifted.rolling(w).std().astype(np.float32)
        features.append(std_col)
    # Time features
    df["week"] = (df["epiweek"] % 100).astype(np.int16)
    df["month"] = pd.to_datetime(df["date"]).dt.month.astype(np.int16)
    df["sin_week"] = np.sin(2 * np.pi * df["week"] / 52.17857).astype(np.float32)
    df["cos_week"] = np.cos(2 * np.pi * df["week"] / 52.17857).astype(np.float32)
    features.extend(["week", "month", "sin_week", "cos_week"])
    
    # Calculate historical weekly baseline in place
    train_mask = df["year"] <= 2021
    hist_mean_df = df[train_mask].groupby(["geocode", "week"])[target].mean()
    df[f"hist_mean_{target}"] = df.set_index(["geocode", "week"]).index.map(hist_mean_df)
    overall_week_mean = df[train_mask].groupby("week")[target].mean().to_dict()
    df[f"hist_mean_{target}"] = df[f"hist_mean_{target}"].fillna(df["week"].map(overall_week_mean))
    df[f"hist_mean_{target}"] = df[f"hist_mean_{target}"].fillna(df[target].mean()).astype(np.float32)
    features.append(f"hist_mean_{target}")
    
    return features

def generate_dengue_features(df):
    # Lags
    df["lag_1_incidence"] = df.groupby("geocode")["incidence_rate"].shift(1).astype(np.float32)
    df["lag_2_incidence"] = df.groupby("geocode")["incidence_rate"].shift(2).astype(np.float32)
    
    # Diff
    df["diff_lag_1"] = df["lag_1_incidence"] - df["lag_2_incidence"]
    
    # Climate lags
    df["lag_4_humid"] = df.groupby("geocode")["rel_humid_med"].shift(4).astype(np.float32)
    df["lag_2_rain"] = df.groupby("geocode")["precip_tot"].shift(2).astype(np.float32)
    
    # Cases change
    df["cases_per_week_change"] = (
        df.groupby("geocode")["cases"].shift(1) - df.groupby("geocode")["cases"].shift(2)
    ).astype(np.float32)
    
    # Calculate historical weekly baseline for dengue in place
    df["week"] = (df["epiweek"] % 100).astype(np.int16)
    train_mask = df["year"] <= 2021
    hist_mean_inc = df[train_mask].groupby(["geocode", "week"])["incidence_rate"].mean()
    df["hist_mean_incidence"] = df.set_index(["geocode", "week"]).index.map(hist_mean_inc)
    overall_week_mean = df[train_mask].groupby("week")["incidence_rate"].mean().to_dict()
    df["hist_mean_incidence"] = df["hist_mean_incidence"].fillna(df["week"].map(overall_week_mean))
    df["hist_mean_incidence"] = df["hist_mean_incidence"].fillna(df["incidence_rate"].mean()).astype(np.float32)
    
    return TOP_FEATURES

def generate_visualizations():
    print("Generating Visualizations...")
    
    # Paths
    hist_csv = "final_brazil_dengue.csv"
    if not os.path.exists(hist_csv):
        hist_csv = os.path.join("data", "final_brazil_dengue.csv")
    climate_pred_csv = "final/outputs/csv/municipality_climate_2025_2029.csv"
    dengue_pred_csv = "final/outputs/csv/municipality_dengue_2025_2029.csv"
    zone_pred_csv = "final/outputs/csv/zone_dengue_2025_2029.csv"
    
    # 1. Load History (last few years to keep memory low, but enough for graphs)
    print("Loading historical data for context...")
    hist_df = pd.read_csv(hist_csv, usecols=["date", "year", "epiweek", "geocode", "uf", "climate_zone", "population", "cases", "incidence_rate", "temp_med", "temp_min", "temp_max", "precip_tot", "rel_humid_med", "pressure_med", "rainy_days"])
    hist_df = hist_df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    
    # Load Forecasts
    print("Loading forecasted data...")
    clim_fore = pd.read_csv(climate_pred_csv)
    clim_fore["date"] = pd.to_datetime(clim_fore["date"])
    
    deng_fore = pd.read_csv(dengue_pred_csv)
    deng_fore["date"] = pd.to_datetime(deng_fore["date"])
    
    zone_fore = pd.read_csv(zone_pred_csv)
    zone_fore["date"] = pd.to_datetime(zone_fore["date"])
    
    # ----------------------------------------------------
    # PART 1: CLIMATE VALIDATION DIAGNOSTICS
    # ----------------------------------------------------
    print("Generating Climate Validation Plots...")
    # Generate validation predictions for temp_med as a representative variable
    target = "temp_med"
    model_path = f"climate/models/climate_{target}.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        sub_df = hist_df[["date", "year", "epiweek", "geocode", target]].copy()
        features = generate_features_for_target(sub_df, target)
        
        # Validation split
        val_df = sub_df[(sub_df["year"] >= 2022) & (sub_df["year"] <= 2024)].dropna(subset=features)
        if len(val_df) > 0:
            val_preds = model.predict(val_df[features])
            actuals = val_df[target].values
            
            # Scatter Plot: Actual vs Predicted
            plt.figure(figsize=(8, 8))
            plt.scatter(actuals, val_preds, alpha=0.1, color="#1f77b4")
            plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], "r--", lw=2)
            plt.title(f"Climate Validation: Actual vs Predicted ({target})", pad=15)
            plt.xlabel("Actual Value")
            plt.ylabel("Predicted Value")
            plt.tight_layout()
            plt.savefig("final/outputs/graphs/climate_val_actual_vs_pred.png", dpi=150)
            plt.close()
            
            # Residual Plot
            residuals = actuals - val_preds
            plt.figure(figsize=(10, 6))
            plt.scatter(val_preds, residuals, alpha=0.1, color="#d62728")
            plt.axhline(y=0, color="k", linestyle="--", lw=1.5)
            plt.title(f"Climate Validation Residuals ({target})", pad=15)
            plt.xlabel("Predicted Value")
            plt.ylabel("Residual (Actual - Predicted)")
            plt.tight_layout()
            plt.savefig("final/outputs/graphs/climate_val_residuals.png", dpi=150)
            plt.close()
            
        del sub_df, val_df, model
        gc.collect()

    # ----------------------------------------------------
    # PART 2: DENGUE VALIDATION DIAGNOSTICS
    # ----------------------------------------------------
    # ----------------------------------------------------
    # PART 2: DENGUE VALIDATION DIAGNOSTICS (LightGBM)
    # ----------------------------------------------------
    print("Generating Dengue Validation Plots...")
    models = {}
    for zone in range(1, 7):
        m_path = f"dengue/models/dengue_model_zone_{zone}.joblib"
        if os.path.exists(m_path):
            models[float(zone)] = joblib.load(m_path)
            
    if len(models) == 6:
        # Construct the same features as train_dengue.py
        val_df = hist_df.copy()
        
        # 1. Downcast and generate time features
        for col in val_df.columns:
            if val_df[col].dtype == np.float64:
                val_df[col] = val_df[col].astype(np.float32)
            elif val_df[col].dtype == np.int64:
                val_df[col] = val_df[col].astype(np.int32)
        val_df["week"] = (val_df["epiweek"] % 100).astype(np.int16)
        val_df["month"] = val_df["date"].dt.month.astype(np.int16)
        
        # 2. Lags
        val_df["lag_1_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(1).astype(np.float32)
        val_df["lag_2_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(2).astype(np.float32)
        val_df["lag_52_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(52).astype(np.float32)
        val_df["lag_104_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(104).astype(np.float32)
        val_df["diff_lag_1"] = val_df["lag_1_incidence"] - val_df["lag_2_incidence"]
        
        # 3. Immunity index
        val_df["rolling_cases_52w"] = val_df.groupby("geocode")["cases"].shift(1).rolling(window=52, min_periods=1).sum().reset_index(level=0, drop=True).astype(np.float32)
        val_df["cum_incidence_52w"] = (val_df["rolling_cases_52w"] / val_df["population"]) * 100000.0
        
        # 4. Baselines (<= 2022)
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
        
        # Filter validation set (2022-2024)
        features = [
            "hist_mean_incidence", "population", "lag_1_incidence", "diff_lag_1",
            "lag_2_incidence", "lag_52_incidence", "lag_104_incidence", "cum_incidence_52w",
            "temp_med", "temp_anomaly", "precip_tot", "precip_anomaly"
        ]
        
        val_slice = val_df[(val_df["year"] >= 2022) & (val_df["year"] <= 2024)].dropna(subset=features)
        
        if len(val_slice) > 0:
            # Predict using zone-wise models
            preds_diff = np.zeros(len(val_slice), dtype=np.float32)
            for zone in range(1, 7):
                zone_mask = val_slice["climate_zone"] == float(zone)
                if np.any(zone_mask):
                    preds_diff[zone_mask] = models[float(zone)].predict(val_slice[features][zone_mask])
            # Reconstruct prediction (using standard 0.45 validation scale)
            preds_inc = val_slice["lag_1_incidence"].values + 0.45 * preds_diff
            preds_inc = np.clip(preds_inc, 0, None)
            
            val_slice["predicted_incidence"] = preds_inc
            val_slice["predicted_cases"] = (val_slice["predicted_incidence"] * val_slice["population"] / 100000.0).round()
            
            # Scatter Plot
            plt.figure(figsize=(8, 8))
            plt.scatter(val_slice["incidence_rate"], preds_inc, alpha=0.1, color="#2ca02c")
            plt.plot([val_slice["incidence_rate"].min(), val_slice["incidence_rate"].max()], 
                     [val_slice["incidence_rate"].min(), val_slice["incidence_rate"].max()], "r--", lw=2)
            plt.title("Dengue Validation: Actual vs Predicted Incidence Rate (LightGBM)", pad=15)
            plt.xlabel("Actual Incidence Rate")
            plt.ylabel("Predicted Incidence Rate")
            plt.yscale("symlog")
            plt.xscale("symlog")
            plt.tight_layout()
            plt.savefig("final/outputs/graphs/dengue_val_actual_vs_pred.png", dpi=150)
            plt.close()
            
            # Residual Plot
            residuals = val_slice["incidence_rate"] - preds_inc
            plt.figure(figsize=(10, 6))
            plt.scatter(preds_inc, residuals, alpha=0.1, color="#d62728")
            plt.axhline(y=0, color="k", linestyle="--", lw=1.5)
            plt.title("Dengue Validation Residuals (LightGBM)", pad=15)
            plt.xlabel("Predicted Incidence Rate")
            plt.ylabel("Residual (Actual - Predicted)")
            plt.xscale("symlog")
            plt.yscale("symlog")
            plt.tight_layout()
            plt.savefig("final/outputs/graphs/dengue_val_residuals.png", dpi=150)
            plt.close()
            
            # Time-Series Validation Curves per Zone (Actual vs Pred)
            print("Generating Dengue Time-Series Validation Curves per Zone (2022-2024)...")
            zone_pop_dict = (
                val_slice[val_slice["date"].dt.year == 2024]
                .groupby(["geocode", "climate_zone"])["population"].first()
                .reset_index()
                .groupby("climate_zone")["population"].sum()
                .to_dict()
            )
            
            for zone in sorted(val_slice["climate_zone"].unique()):
                zg = val_slice[val_slice["climate_zone"] == zone]
                
                # Group by date to aggregate at zone level
                zg_grouped = zg.groupby("date").agg({
                    "cases": "sum",
                    "predicted_cases": "sum",
                    "population": "sum"
                }).reset_index().sort_values("date")
                
                z_pop = zone_pop_dict.get(zone, zg_grouped["population"].iloc[0])
                zg_grouped["actual_incidence"] = (zg_grouped["cases"] / z_pop) * 100000.0
                zg_grouped["predicted_incidence"] = (zg_grouped["predicted_cases"] / z_pop) * 100000.0
                
                # Smooth predicted curve slightly for visualization (rolling 2-week mean)
                zg_grouped["predicted_incidence_smooth"] = zg_grouped["predicted_incidence"].rolling(2, min_periods=1).mean()
                
                plt.figure(figsize=(14, 6))
                plt.plot(zg_grouped["date"], zg_grouped["actual_incidence"], label="Actual", color="#1f77b4", lw=2)
                plt.plot(zg_grouped["date"], zg_grouped["predicted_incidence_smooth"], label="Predicted", color="#ff7f0e", lw=2)
                
                plt.title(f"LightGBM Validation - Climate Zone {int(zone)} (Actual vs Predicted)", fontsize=14, fontweight="bold", pad=15)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Incidence Rate (per 100k)", fontsize=12)
                plt.legend(loc="upper right", fontsize=10)
                plt.tight_layout()
                plt.savefig(f"final/outputs/graphs/dengue_val_time_series_zone_{int(zone)}.png", dpi=150)
                plt.close()
                print(f"  Saved time-series validation graph for Zone {int(zone)}")
                
        del val_df, val_slice, models
        gc.collect()

    # ----------------------------------------------------
    # PART 3: CLIMATE FORECAST VS HISTORY COMPARISON
    # ----------------------------------------------------
    print("Generating Climate Forecast plots with history...")
    # Overall weekly averages
    # Slice history from 2020 onwards for visual clarity
    hist_recent = hist_df[hist_df["year"] >= 2020]
    
    # 1. Weekly Average Temperature
    hist_avg_temp = hist_recent.groupby("date")["temp_med"].mean()
    fore_avg_temp = clim_fore.groupby("date")["temp_med"].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(hist_avg_temp.index, hist_avg_temp.values, label="Historical (2020-2024)", color="#1f77b4", lw=2)
    plt.plot(fore_avg_temp.index, fore_avg_temp.values, label="Forecast (2025-2029)", color="#ff7f0e", lw=2, linestyle="--")
    plt.title("Brazil Average Temperature (temp_med) - History vs Forecast", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("final/outputs/graphs/climate_forecast_temp_med_overall.png", dpi=150)
    plt.close()
    
    # 2. Weekly Total Rainfall
    # precip_tot is total rainfall. Summing across municipalities doesn't make physical sense, so we take mean of municipality totals.
    hist_avg_rain = hist_recent.groupby("date")["precip_tot"].mean()
    fore_avg_rain = clim_fore.groupby("date")["precip_tot"].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(hist_avg_rain.index, hist_avg_rain.values, label="Historical (2020-2024)", color="#1f77b4", lw=2)
    plt.plot(fore_avg_rain.index, fore_avg_rain.values, label="Forecast (2025-2029)", color="#2ca02c", lw=2, linestyle="--")
    plt.title("Brazil Average Weekly Total Precipitation (precip_tot) - History vs Forecast", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Precipitation (mm)", fontsize=12)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("final/outputs/graphs/climate_forecast_precip_tot_overall.png", dpi=150)
    plt.close()

    # 3. Sample Municipalities Climate
    sample_geocodes = [1100015, 3550308] # Rondonia sample, São Paulo city
    for geo in sample_geocodes:
        hist_geo = hist_recent[hist_recent["geocode"] == geo].sort_values("date")
        fore_geo = clim_fore[clim_fore["geocode"] == geo].sort_values("date")
        
        if len(hist_geo) > 0 and len(fore_geo) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(hist_geo["date"], hist_geo["temp_med"], label="Historical", color="#1f77b4", alpha=0.8)
            plt.plot(fore_geo["date"], fore_geo["temp_med"], label="Forecast", color="#ff7f0e", linestyle="--")
            plt.title(f"Temperature Forecast for Municipality {geo} - History vs Forecast", fontsize=14, fontweight="bold", pad=15)
            plt.xlabel("Date")
            plt.ylabel("Temperature (°C)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"final/outputs/graphs/climate_forecast_mun_{geo}.png", dpi=150)
            plt.close()

    # ----------------------------------------------------
    # PART 4: DENGUE FORECAST VS HISTORY COMPARISON
    # ----------------------------------------------------
    print("Generating Dengue Forecast plots with history...")
    
    # 1. Sample Municipalities Dengue
    for geo in sample_geocodes:
        hist_geo = hist_recent[hist_recent["geocode"] == geo].sort_values("date")
        fore_geo = deng_fore[deng_fore["geocode"] == geo].sort_values("date")
        
        if len(hist_geo) > 0 and len(fore_geo) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(hist_geo["date"], hist_geo["incidence_rate"], label="Historical Incidence", color="#1f77b4", lw=2)
            plt.plot(fore_geo["date"], fore_geo["incidence_rate"], label="Forecast Incidence", color="#d62728", lw=2, linestyle="--")
            plt.title(f"Dengue Incidence Rate for Municipality {geo} - History vs Forecast", fontsize=14, fontweight="bold", pad=15)
            plt.xlabel("Date")
            plt.ylabel("Incidence Rate (per 100k)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"final/outputs/graphs/dengue_forecast_mun_{geo}.png", dpi=150)
            plt.close()
            
    # 2. Zone aggregated graphs
    # To compute historical climate zone averages:
    # Historical cases and populations aggregated by zone and date
    # Historical zone population = sum of unique state populations per zone
    # Correct zone populations: sum of ALL municipality populations in each zone
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
                "date": date_val,
                "climate_zone": zone,
                "cases": z_cases,
                "incidence_rate": z_inc
            })
        return pd.DataFrame(records)
        
    print("Aggregating historical dengue data by climate zone...")
    hist_zone_df = hist_recent.groupby("date", group_keys=False).apply(agg_hist_zones).reset_index(drop=True)
    
    # Plot history + forecast for each climate zone
    for zone in sorted(zone_fore["climate_zone"].unique()):
        hz = hist_zone_df[hist_zone_df["climate_zone"] == zone].sort_values("date")
        fz = zone_fore[zone_fore["climate_zone"] == zone].sort_values("date")
        
        hist_peak = hz["incidence_rate"].max()
        fore_peak = fz["incidence_rate"].max()
        
        # Compute ±1 std confidence band for forecast (rolling 4-week std)
        fore_std = fz["incidence_rate"].rolling(4, center=True, min_periods=1).std().fillna(0)
        fore_upper = fz["incidence_rate"] + fore_std
        fore_lower = (fz["incidence_rate"] - fore_std).clip(lower=0)
        
        fig, ax1 = plt.subplots(figsize=(14, 6))
        plt.style.use("seaborn-v0_8-whitegrid")
        
        # Use dual Y-axis only when scales differ by more than 3× in either direction
        use_dual = (hist_peak > 0) and (fore_peak > 0) and (max(hist_peak / fore_peak, fore_peak / hist_peak) > 3.0)
        
        if use_dual:
            ax2 = ax1.twinx()
            ax1.plot(hz["date"], hz["incidence_rate"], label="Historical", color="#1f77b4", lw=2.5)
            ax2.plot(fz["date"], fz["incidence_rate"], label="Forecast (right axis)", color="#ff7f0e", lw=2.5, linestyle="--")
            ax2.fill_between(fz["date"], fore_lower, fore_upper, color="#ff7f0e", alpha=0.18, label="Forecast ±1σ")
            ax1.set_ylabel("Historical Incidence Rate (per 100k)", fontsize=11, color="#1f77b4")
            ax2.set_ylabel("Forecast Incidence Rate (per 100k)", fontsize=11, color="#ff7f0e")
            ax1.tick_params(axis="y", labelcolor="#1f77b4")
            ax2.tick_params(axis="y", labelcolor="#ff7f0e")
            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
            ax1.set_title(
                f"Dengue Incidence Rate - Climate Zone {zone} - History vs Forecast\n"
                f"(Note: dual Y-axis — historical peak {hist_peak:.0f}, forecast peak {fore_peak:.0f} per 100k)",
                fontsize=13, fontweight="bold", pad=12
            )
        else:
            ax1.plot(hz["date"], hz["incidence_rate"], label="Historical", color="#1f77b4", lw=2.5)
            ax1.plot(fz["date"], fz["incidence_rate"], label="Forecast", color="#ff7f0e", lw=2.5, linestyle="--")
            ax1.fill_between(fz["date"], fore_lower, fore_upper, color="#ff7f0e", alpha=0.18, label="Forecast ±1σ")
            ax1.set_ylabel("Incidence Rate (per 100k)", fontsize=11)
            ax1.legend(fontsize=10)
            ax1.set_title(
                f"Dengue Incidence Rate - Climate Zone {zone} - History vs Forecast",
                fontsize=14, fontweight="bold", pad=15
            )
        
        # Add vertical line at forecast start
        forecast_start = fz["date"].min()
        ax1.axvline(x=forecast_start, color="gray", linestyle=":", lw=1.5, alpha=0.7)
        ax1.text(forecast_start, ax1.get_ylim()[1] * 0.95, "  Forecast →", fontsize=9, color="gray", va="top")
        
        ax1.set_xlabel("Date", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"final/outputs/graphs/dengue_forecast_zone_{zone}.png", dpi=150)
        plt.close()
        
    # Combined zone forecast line plot
    plt.figure(figsize=(14, 7))
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    for i, zone in enumerate(sorted(zone_fore["climate_zone"].unique())):
        fz = zone_fore[zone_fore["climate_zone"] == zone].sort_values("date")
        c = colors[i % len(colors)]
        plt.plot(fz["date"], fz["incidence_rate"], label=f"Zone {zone}", lw=2, color=c)
        fstd = fz["incidence_rate"].rolling(4, center=True, min_periods=1).std().fillna(0)
        plt.fill_between(fz["date"], (fz["incidence_rate"]-fstd).clip(0), fz["incidence_rate"]+fstd, alpha=0.10, color=c)
    plt.title("Dengue Incidence Forecast by Climate Zone (2025–2029)", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Incidence Rate (per 100k)", fontsize=12)
    plt.legend(title="Climate Zone", fontsize=10)
    plt.tight_layout()
    plt.savefig("final/outputs/graphs/dengue_forecast_combined_zones.png", dpi=150)
    plt.close()
    
    print("Visualizations generated successfully. All plots saved to final/outputs/graphs/")

if __name__ == "__main__":
    generate_visualizations()
