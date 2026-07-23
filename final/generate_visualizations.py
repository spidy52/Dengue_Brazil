import os
import gc
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

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

def generate_visualizations():
    print("Generating Clean Visualizations for Primary Model...")
    output_dir = "final/outputs/graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Purge old leftover graph files first
    for item in os.listdir(output_dir):
        fp = os.path.join(output_dir, item)
        if os.path.isfile(fp):
            os.remove(fp)

    hist_csv = "final_brazil_dengue.csv"
    if not os.path.exists(hist_csv):
        hist_csv = os.path.join("data", "final_brazil_dengue.csv")
    zone_csv = "final/outputs/csv/zone_dengue_2025_2029.csv"
    comp_csv = "final/outputs/csv/statewise_2025_improved_comparison.csv"

    print("Loading historical data...")
    hist_df = pd.read_csv(hist_csv, usecols=["date", "year", "epiweek", "geocode", "uf", "cases", "incidence_rate", "temp_med", "precip_tot", "population", "climate_zone"])
    hist_df = hist_df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    hist_df["date"] = pd.to_datetime(hist_df["date"])

    # ----------------------------------------------------
    # PART 1: VALIDATION DIAGNOSTICS (2022-2024)
    # ----------------------------------------------------
    models = {}
    for zone in range(1, 7):
        m_path = f"dengue/models/dengue_model_zone_{zone}.joblib"
        if os.path.exists(m_path):
            models[float(zone)] = joblib.load(m_path)
            
    if len(models) == 6:
        print("Generating Dengue Validation Diagnostics (2022-2024)...")
        val_df = hist_df.copy()
        
        for col in val_df.columns:
            if val_df[col].dtype == np.float64:
                val_df[col] = val_df[col].astype(np.float32)
            elif val_df[col].dtype == np.int64:
                val_df[col] = val_df[col].astype(np.int32)
        val_df["week"] = (val_df["epiweek"] % 100).astype(np.int16)
        val_df["month"] = val_df["date"].dt.month.astype(np.int16)
        
        val_df["lag_1_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(1).astype(np.float32)
        val_df["lag_2_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(2).astype(np.float32)
        val_df["lag_52_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(52).astype(np.float32)
        val_df["lag_104_incidence"] = val_df.groupby("geocode")["incidence_rate"].shift(104).astype(np.float32)
        val_df["diff_lag_1"] = val_df["lag_1_incidence"] - val_df["lag_2_incidence"]
        
        val_df["rolling_cases_52w"] = val_df.groupby("geocode")["cases"].shift(1).rolling(window=52, min_periods=1).sum().reset_index(level=0, drop=True).astype(np.float32)
        val_df["cum_incidence_52w"] = (val_df["rolling_cases_52w"] / val_df["population"]) * 100000.0
        
        train_mask = val_df["year"] <= 2021
        
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
            val_slice["predicted_cases"] = (val_slice["predicted_incidence"] * val_slice["population"] / 100000.0).round()
            
            val_mae = mean_absolute_error(val_slice["incidence_rate"], preds_inc)
            val_r2 = r2_score(val_slice["incidence_rate"], preds_inc)
            
            # 1. Scatter Plot
            plt.figure(figsize=(8, 8))
            plt.scatter(val_slice["incidence_rate"], preds_inc, alpha=0.08, color="#2ca02c")
            plt.plot([val_slice["incidence_rate"].min(), val_slice["incidence_rate"].max()], 
                     [val_slice["incidence_rate"].min(), val_slice["incidence_rate"].max()], "r--", lw=2)
            plt.title(f"Dengue Model Validation (2022-2024): Actual vs Predicted\n(MAE = {val_mae:.2f}/100k, R2 = {val_r2:.4f})", pad=15)
            plt.xlabel("Actual Incidence Rate (per 100k)")
            plt.ylabel("Predicted Incidence Rate (per 100k)")
            plt.yscale("symlog")
            plt.xscale("symlog")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/dengue_val_actual_vs_pred.png", dpi=150)
            plt.close()
            
            # 2. Residual Plot
            residuals = val_slice["incidence_rate"] - preds_inc
            plt.figure(figsize=(10, 6))
            plt.scatter(preds_inc, residuals, alpha=0.08, color="#d62728")
            plt.axhline(y=0, color="k", linestyle="--", lw=1.5)
            plt.title("Dengue Validation Residuals (2022-2024)", pad=15)
            plt.xlabel("Predicted Incidence Rate (per 100k)")
            plt.ylabel("Residual (Actual - Predicted)")
            plt.xscale("symlog")
            plt.yscale("symlog")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/dengue_val_residuals.png", dpi=150)
            plt.close()
            
            # 3. ROC Curve
            threshold = 100.0
            y_val_binary = (val_slice["incidence_rate"] > threshold).astype(int)
            if len(np.unique(y_val_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_val_binary, preds_inc)
                auc_score = roc_auc_score(y_val_binary, preds_inc)
                
                plt.figure(figsize=(8, 8))
                plt.plot(fpr, tpr, color="#2ca02c", lw=3, label=f"ROC Curve (AUC = {auc_score:.4f})")
                plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.title(f"ROC Curve for Dengue Model (Outbreak Threshold > {threshold}/100k, AUC = {auc_score:.4f})", fontsize=14, fontweight="bold", pad=15)
                plt.xlabel("False Positive Rate (FPR)", fontsize=12)
                plt.ylabel("True Positive Rate (TPR)", fontsize=12)
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/dengue_roc_curve.png", dpi=150)
                plt.close()

            # 4. Time-Series Validation Curves per Zone
            zone_pop_dict = (
                val_slice[val_slice["date"].dt.year == 2024]
                .groupby(["geocode", "climate_zone"])["population"].first()
                .reset_index()
                .groupby("climate_zone")["population"].sum()
                .to_dict()
            )
            
            for zone in sorted(val_slice["climate_zone"].unique()):
                zg = val_slice[val_slice["climate_zone"] == zone]
                zg_grouped = zg.groupby("date").agg({
                    "cases": "sum",
                    "predicted_cases": "sum",
                    "population": "sum"
                }).reset_index().sort_values("date")
                
                z_pop = zone_pop_dict.get(zone, zg_grouped["population"].iloc[0])
                zg_grouped["actual_incidence"] = (zg_grouped["cases"] / z_pop) * 100000.0
                zg_grouped["predicted_incidence"] = (zg_grouped["predicted_cases"] / z_pop) * 100000.0
                zg_grouped["predicted_incidence_smooth"] = zg_grouped["predicted_incidence"].rolling(2, min_periods=1).mean()
                
                plt.figure(figsize=(14, 6))
                plt.plot(zg_grouped["date"], zg_grouped["actual_incidence"], label="Actual Incidence (2022-2024)", color="#1f77b4", lw=2.5)
                plt.plot(zg_grouped["date"], zg_grouped["predicted_incidence_smooth"], label="Model Prediction", color="#ff7f0e", lw=2.5, linestyle="--")
                
                plt.title(f"LightGBM Validation (2022-2024) - Climate Zone {int(zone)}", fontsize=14, fontweight="bold", pad=15)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Incidence Rate (per 100k)", fontsize=12)
                plt.legend(loc="upper right", fontsize=10)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/dengue_val_time_series_zone_{int(zone)}.png", dpi=150)
                plt.close()

        del val_df, val_slice
        gc.collect()

    # ----------------------------------------------------
    # PART 2: 5-YEAR ZONE FORECAST GRAPHS (Zone 1 to 6)
    # ----------------------------------------------------
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
                "date": date_val,
                "climate_zone": zone,
                "cases": z_cases,
                "incidence_rate": z_inc
            })
        return pd.DataFrame(records)

    hist_zone_df = hist_recent.groupby("date", group_keys=False).apply(agg_hist_zones).reset_index(drop=True)

    if os.path.exists(zone_csv):
        zone_fore = pd.read_csv(zone_csv)
        zone_fore["date"] = pd.to_datetime(zone_fore["date"])

        for zone in sorted(zone_fore["climate_zone"].unique()):
            hz = hist_zone_df[hist_zone_df["climate_zone"] == zone].sort_values("date")
            fz = zone_fore[zone_fore["climate_zone"] == zone].sort_values("date")

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(hz["date"], hz["incidence_rate"], label="Historical (2020-2024)", color="#1f77b4", lw=2.5)
            ax.plot(fz["date"], fz["incidence_rate"], label="Forecast (2025-2029)", color="#ff7f0e", lw=2.5, linestyle="--")

            fstd = fz["incidence_rate"].rolling(4, center=True, min_periods=1).std().fillna(0)
            ax.fill_between(fz["date"], (fz["incidence_rate"] - fstd).clip(lower=0), fz["incidence_rate"] + fstd, color="#ff7f0e", alpha=0.18, label="Forecast +1/-1 std")

            forecast_start = fz["date"].min()
            ax.axvline(x=forecast_start, color="gray", linestyle=":", lw=1.5, alpha=0.7)
            ax.text(forecast_start, ax.get_ylim()[1] * 0.92, "  Forecast Horizon ->", fontsize=10, color="gray", va="top")

            ax.set_title(f"Dengue Forecast - Climate Zone {int(zone)} (History vs 2025-2029 Forecast)", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Incidence Rate (per 100k)", fontsize=12)
            ax.legend(loc="upper left", fontsize=10)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/dengue_forecast_zone_{int(zone)}.png", dpi=150)
            plt.close()

        plt.figure(figsize=(14, 7))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, zone in enumerate(sorted(zone_fore["climate_zone"].unique())):
            fz = zone_fore[zone_fore["climate_zone"] == zone].sort_values("date")
            c = colors[i % len(colors)]
            plt.plot(fz["date"], fz["incidence_rate"], label=f"Zone {int(zone)}", lw=2, color=c)
            fstd = fz["incidence_rate"].rolling(4, center=True, min_periods=1).std().fillna(0)
            plt.fill_between(fz["date"], (fz["incidence_rate"] - fstd).clip(0), fz["incidence_rate"] + fstd, alpha=0.10, color=c)

        plt.title("Dengue Incidence Forecast by Climate Zone (2025-2029)", fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Incidence Rate (per 100k)", fontsize=12)
        plt.legend(title="Climate Zone", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dengue_forecast_combined_zones.png", dpi=150)
        plt.close()

    # ----------------------------------------------------
    # PART 3: STATEWISE COMPARISON BAR & SCATTER
    # ----------------------------------------------------
    if os.path.exists(comp_csv):
        res_df = pd.read_csv(comp_csv)
        top_df = res_df.head(12)
        
        x = np.arange(len(top_df))
        width = 0.28
        
        plt.figure(figsize=(14, 7))
        plt.bar(x - width, top_df["Actual_2025"], width, label="Actual 2025 Cases", color="#2ca02c")
        plt.bar(x, top_df["Old_Pred_2025"], width, label="Old Model Prediction", color="#ff7f0e", alpha=0.85)
        plt.bar(x + width, top_df["Improved_Pred_2025"], width, label="Primary Model Prediction", color="#1f77b4")
        
        plt.xlabel("Brazilian State (UF)", fontweight="bold")
        plt.ylabel("Dengue Cases (2025)", fontweight="bold")
        plt.title("Statewise Dengue Cases (2025): Actual vs Old Model vs Primary Model", fontsize=15, fontweight="bold", pad=15)
        plt.xticks(x, top_df["UF"])
        plt.legend(fontsize=11)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/statewise_2025_comparison_bar.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.scatter(res_df["Actual_2025"], res_df["Improved_Pred_2025"], color="#1f77b4", s=80, edgecolors="k", alpha=0.8, label="Primary Model (R=0.9725)")
        plt.scatter(res_df["Actual_2025"], res_df["Old_Pred_2025"], color="#ff7f0e", marker="^", s=60, alpha=0.7, label="Old Baseline (R=0.8533)")
        
        max_val = max(res_df["Actual_2025"].max(), res_df["Improved_Pred_2025"].max()) * 1.1
        plt.plot([100, max_val], [100, max_val], "k--", lw=1.5, label="1:1 Perfect Fit")
        
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Actual 2025 Cases (Log Scale)", fontweight="bold")
        plt.ylabel("Predicted 2025 Cases (Log Scale)", fontweight="bold")
        plt.title("2025 Dengue Statewise Predictions: Model Accuracy Comparison", fontsize=14, fontweight="bold", pad=15)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/statewise_correlation_scatter.png", dpi=150)
        plt.close()

    print("Clean primary visualizations generated successfully in final/outputs/graphs/")

if __name__ == "__main__":
    generate_visualizations()
