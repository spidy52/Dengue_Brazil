import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.linewidth": 2.2,
    "axes.edgecolor": "#000000",
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size": 6,
    "ytick.major.size": 6
})

DYNAMIC_FEATURES = [
    "log_lag1",
    "lag_1_inc",
    "diff_lag_1",
    "lag_2_inc",
    "lag_3_inc",
    "lag_4_inc",
    "lag_52_inc",
    "roll_mean_4",
    "roll_mean_8",
    "roll_std_4",
    "vector_activity_index",
    "breeding_index",
    "temp_med",
    "precip_tot",
    "rel_humid_med",
    "population",
    "muni_mean_inc",
    "muni_max_inc",
    "sin_week",
    "cos_week",
    "month"
]

FEATURE_LABELS = {
    "log_lag1": "Log(Lag-1 Incidence)",
    "lag_1_inc": "Lag-1 Incidence Rate",
    "diff_lag_1": "1-Week Incidence Difference",
    "lag_2_inc": "Lag-2 Incidence Rate",
    "lag_3_inc": "Lag-3 Incidence Rate",
    "lag_4_inc": "Lag-4 Incidence Rate",
    "lag_52_inc": "52-Week Seasonal Lag",
    "roll_mean_4": "4-Week Moving Average",
    "roll_mean_8": "8-Week Moving Average",
    "roll_std_4": "4-Week Moving Std Dev",
    "vector_activity_index": "Vector Activity Index",
    "breeding_index": "Breeding Index",
    "temp_med": "Median Temperature (°C)",
    "precip_tot": "Total Precipitation (mm)",
    "rel_humid_med": "Relative Humidity (%)",
    "population": "Municipal Population",
    "muni_mean_inc": "Historical Baseline Mean",
    "muni_max_inc": "Historical Peak Incidence",
    "sin_week": "Sinusoidal Week Component",
    "cos_week": "Cosinusoidal Week Component",
    "month": "Calendar Month"
}

ZONE_NAMES = {
    1: "Zone 1 (Equatorial Amazon)",
    2: "Zone 2 (Cerrado North)",
    3: "Zone 3 (Semi-Arid NE)",
    4: "Zone 4 (Central-West)",
    5: "Zone 5 (Southeast Core)",
    6: "Zone 6 (Southern Temperate)"
}

def generate_validation_plots():
    print("=========================================================")
    print("=== GENERATING PURE CLIMATE-DRIVEN VALIDATION FIGURES ===")
    print("=========================================================")
    
    csv_path = "final_brazil_dengue.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "final_brazil_dengue.csv")
        
    cols = [
        "date", "year", "epiweek", "geocode", "uf", "cases", "incidence_rate",
        "temp_med", "precip_tot", "rel_humid_med", "population", "climate_zone"
    ]
    df = pd.read_csv(csv_path, usecols=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["geocode", "date"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    
    # Feature engineering
    df["week"] = (df["epiweek"] % 100).astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int16)
    df["sin_week"] = np.sin(2 * np.pi * df["week"] / 52.17857).astype(np.float32)
    df["cos_week"] = np.cos(2 * np.pi * df["week"] / 52.17857).astype(np.float32)
    
    df["lag_1_inc"] = df.groupby("geocode")["incidence_rate"].shift(1).astype(np.float32)
    df["lag_2_inc"] = df.groupby("geocode")["incidence_rate"].shift(2).astype(np.float32)
    df["lag_3_inc"] = df.groupby("geocode")["incidence_rate"].shift(3).astype(np.float32)
    df["lag_4_inc"] = df.groupby("geocode")["incidence_rate"].shift(4).astype(np.float32)
    df["lag_52_inc"] = df.groupby("geocode")["incidence_rate"].shift(52).astype(np.float32)
    df["diff_lag_1"] = df["lag_1_inc"] - df["lag_2_inc"]
    
    df["roll_mean_4"] = df.groupby("geocode")["lag_1_inc"].rolling(4).mean().reset_index(level=0, drop=True).astype(np.float32)
    df["roll_mean_8"] = df.groupby("geocode")["lag_1_inc"].rolling(8).mean().reset_index(level=0, drop=True).astype(np.float32)
    df["roll_std_4"] = df.groupby("geocode")["lag_1_inc"].rolling(4).std().reset_index(level=0, drop=True).astype(np.float32)
    
    df["vector_activity_index"] = (df["temp_med"] * (df["rel_humid_med"] / 100.0)).astype(np.float32)
    df["breeding_index"] = (df["precip_tot"] * (df["temp_med"] / 30.0)).astype(np.float32)
    
    train_mask = df["year"] <= 2022
    muni_stats = df[train_mask].groupby("geocode")["incidence_rate"].agg(["mean", "max"]).reset_index().rename(columns={"mean": "muni_mean_inc", "max": "muni_max_inc"})
    df = df.merge(muni_stats, on="geocode", how="left")
    df["muni_mean_inc"] = df["muni_mean_inc"].fillna(df["incidence_rate"].mean()).astype(np.float32)
    df["muni_max_inc"] = df["muni_max_inc"].fillna(df["incidence_rate"].max()).astype(np.float32)
    
    df["log_inc"] = np.log1p(df["incidence_rate"]).astype(np.float32)
    df["log_lag1"] = np.log1p(df["lag_1_inc"]).astype(np.float32)
    
    df_clean = df.dropna(subset=DYNAMIC_FEATURES + ["incidence_rate"]).reset_index(drop=True)
    val_mask = (df_clean["date"] >= "2022-01-01") & (df_clean["date"] <= "2024-06-30")
    df_val = df_clean[val_mask].copy()
    
    # Load trained models
    models = {}
    for z in range(1, 7):
        m_path = f"dengue/models/dengue_model_zone_{z}.joblib"
        if os.path.exists(m_path):
            models[z] = joblib.load(m_path)
            
    df_val["pred_log"] = 0.0
    for z in range(1, 7):
        z_mask = df_val["climate_zone"] == float(z)
        if np.any(z_mask) and z in models:
            # 100% PURE DIRECT DYNAMIC MODEL INFERENCE
            df_val.loc[z_mask, "pred_log"] = models[z].predict(df_val.loc[z_mask, DYNAMIC_FEATURES])
            
    df_val["pred_inc"] = np.clip(np.expm1(df_val["pred_log"]), 0, None)
    df_val["pred_cases"] = (df_val["pred_inc"] / 100000.0) * df_val["population"]
    
    output_dirs = ["figures", "final/outputs/graphs"]
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        
    zone_weekly = df_val.groupby(["climate_zone", "date"]).apply(
        lambda g: pd.Series({
            "cases": g["cases"].sum(),
            "population": g["population"].sum(),
            "pred_cases": g["pred_cases"].sum(),
            "actual_incidence": (g["cases"].sum() / g["population"].sum()) * 100000.0,
            "predicted_incidence": (g["pred_cases"].sum() / g["population"].sum()) * 100000.0
        })
    ).reset_index()

    # 1. Generate Individual Zone Validation Figures (Pure Dynamic Climate Mode Only)
    for zone_id in range(1, 7):
        z_data = zone_weekly[zone_weekly["climate_zone"] == zone_id].sort_values("date").copy().reset_index(drop=True)
        
        act = z_data["actual_incidence"].values
        pred = z_data["predicted_incidence"].values
        
        r2 = 1 - np.sum((act - pred)**2) / np.sum((act - np.mean(act))**2)
        mae = np.mean(np.abs(act - pred))
        print(f"Pure Climate Validation Figure - {ZONE_NAMES[zone_id]}: R^2 = {r2:.4f}, MAE = {mae:.2f} per 100k")
        
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.plot(z_data["date"], act, label="Actual Incidence (2022-2024)", color="#1f77b4", linewidth=2.0)
        ax.plot(z_data["date"], pred, label="Model Prediction", color="#ff7f0e", linestyle="--", linewidth=2.0)
        
        ax.grid(False)
        ax.set_xlabel("Date", fontsize=11, fontweight="bold", color="#000000", labelpad=8)
        ax.set_ylabel("Incidence Rate (per 100k)", fontsize=11, fontweight="bold", color="#000000", labelpad=8)
        
        # Thick black box axes
        for spine in ax.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(2.0)
            
        ax.tick_params(colors="#000000", labelsize=10, width=1.5, length=5)
        # Place legend in the upper middle of the plot with 2 columns
        ax.legend(frameon=False, loc="upper center", fontsize=10, ncol=2)
        
        plt.tight_layout()
        
        for d in output_dirs:
            png_path = os.path.join(d, f"dengue_validation_zone_{zone_id}.png")
            eps_path = os.path.join(d, f"dengue_validation_zone_{zone_id}.eps")
            pdf_path = os.path.join(d, f"dengue_validation_zone_{zone_id}.pdf")
            plt.savefig(png_path, dpi=600, bbox_inches="tight")
            plt.savefig(eps_path, format="eps", bbox_inches="tight")
            plt.savefig(pdf_path, bbox_inches="tight")
            
        plt.close()

    # 2. Generate 6-Panel Combined Zone Validation Plot (Pure Dynamic Climate Mode, 2x3 Grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), facecolor="white", sharex=True)
    axes = axes.flatten()

    for i, zone_id in enumerate(range(1, 7)):
        ax = axes[i]
        z_data = zone_weekly[zone_weekly["climate_zone"] == zone_id].sort_values("date").copy().reset_index(drop=True)
        
        act = z_data["actual_incidence"].values
        pred = z_data["predicted_incidence"].values
        
        ax.plot(z_data["date"], act, label="Actual Incidence", color="#1f77b4", linewidth=1.8)
        ax.plot(z_data["date"], pred, label="Model Prediction", color="#ff7f0e", linestyle="--", linewidth=1.8)
        
        ax.grid(False)
        ax.text(0.03, 0.90, f"Zone {zone_id}", transform=ax.transAxes, fontsize=9.5, fontweight="bold", ha="left", color="#333333")
        ax.set_ylabel("Incidence Rate (per 100k)", fontsize=8.5, color="#333333")
        if i >= 3:
            ax.set_xlabel("Date", fontsize=9.5, fontweight="bold", color="#333333")
            
        for spine in ax.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(2.0)
            
        ax.tick_params(colors="#333333", labelsize=8.5, width=1.5, length=4)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    plt.tight_layout()
    for d in output_dirs:
        plt.savefig(os.path.join(d, "dengue_validation_combined_zones.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(d, "dengue_validation_combined_zones.eps"), format="eps", bbox_inches="tight")
        plt.savefig(os.path.join(d, "dengue_validation_combined_zones.pdf"), bbox_inches="tight")
    plt.close()

    # 2b. State-Level National Aggregate Validation Plot (Thick Axes & Centered Legend)
    state_weekly = df_val.groupby("date").apply(
        lambda g: pd.Series({
            "cases": g["cases"].sum() / 1000.0,
            "pred_cases": g["pred_cases"].sum() / 1000.0
        })
    ).reset_index().sort_values("date")
    
    act_st = state_weekly["cases"].values
    pred_st = state_weekly["pred_cases"].values
    
    r2_st = 1 - np.sum((act_st - pred_st)**2) / np.sum((act_st - np.mean(act_st))**2)
    pearson_st = np.corrcoef(act_st, pred_st)[0, 1]
    
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.plot(state_weekly["date"], act_st, label="Actual Ground Truth", color="#1f77b4", linewidth=2.0)
    ax.plot(state_weekly["date"], pred_st, label="LightGBM Model Prediction", color="#ff7f0e", linestyle="--", linewidth=2.0)
    
    stats_text = f"State-Level $R^2 = {r2_st:.4f}$\nPearson $r = {pearson_st:.4f}$"
    ax.text(0.04, 0.88, stats_text, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#aaaaaa", alpha=0.9), zorder=10)
    
    ax.grid(False)
    ax.set_xlabel("Validation Date (2023–2024 Holdout Set)", fontsize=11, fontweight="bold", fontstyle="italic", labelpad=8)
    ax.set_ylabel("Weekly Dengue Cases (in Thousands)", fontsize=11, fontweight="bold", fontstyle="italic", labelpad=8)
    
    # Thick black box axes
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(2.0)
        
    ax.tick_params(colors="#000000", labelsize=10, width=1.5, length=5)
    # Legend centered in upper middle
    ax.legend(frameon=False, loc="upper center", fontsize=10, ncol=2)
    
    plt.tight_layout()
    for d in output_dirs:
        plt.savefig(os.path.join(d, "dengue_validation_state_aggregate.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(d, "dengue_validation_state_aggregate.pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(d, "dengue_validation_state_aggregate.eps"), format="eps", bbox_inches="tight")
        plt.savefig(os.path.join(d, "dengue_validation_actual_vs_predicted.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(d, "dengue_validation_actual_vs_predicted.eps"), format="eps", bbox_inches="tight")
    plt.close()

    # 3. Outbreak ROC-AUC Curve
    act_inc = df_val["incidence_rate"].values
    pred_inc = df_val["pred_inc"].values
    threshold = np.percentile(act_inc, 75)
    y_true = (act_inc >= threshold).astype(int)
    
    fpr, tpr, _ = roc_curve(y_true, pred_inc)
    roc_auc = roc_auc_score(y_true, pred_inc)
    
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.plot(fpr, tpr, color="#d62728", lw=2.5, label=f"LightGBM Dengue Model (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Baseline (AUC = 0.50)")
    
    ax.grid(False)
    ax.set_xlabel("False Positive Rate", fontsize=11, color="#333333")
    ax.set_ylabel("True Positive Rate", fontsize=11, color="#333333")
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10, loc="lower right", frameon=False)
    
    plt.tight_layout()
    for d in output_dirs:
        plt.savefig(os.path.join(d, "dengue_outbreak_roc_curve.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(d, "dengue_outbreak_roc_curve.eps"), format="eps", bbox_inches="tight")
    plt.close()

    # 4. Feature Importance Plot
    importances = np.zeros(len(DYNAMIC_FEATURES))
    for z in range(1, 7):
        if z in models:
            importances += models[z].feature_importances_
    importances /= max(len(models), 1)
    
    feat_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in DYNAMIC_FEATURES],
        "Importance": importances
    }).sort_values("Importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2ca02c", edgecolor="none", height=0.65)
    
    ax.grid(False)
    ax.set_xlabel("Mean Feature Importance (Split Count across Macro-Zones)", fontsize=11, color="#333333")
    ax.tick_params(axis="both", labelsize=10)
    
    plt.tight_layout()
    for d in output_dirs:
        plt.savefig(os.path.join(d, "dengue_feature_importance.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(d, "dengue_feature_importance.eps"), format="eps", bbox_inches="tight")
    plt.close()
    
    print("\nPure Climate-Driven Validation Plots generated successfully!")

if __name__ == "__main__":
    generate_validation_plots()
