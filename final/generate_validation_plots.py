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
from sklearn.metrics import roc_curve, roc_auc_score, r2_score, mean_absolute_error, auc
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "figure.dpi": 600
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
    "vector_activity_index": "Vector Activity Index (Temp x RH)",
    "breeding_index": "Breeding Index (Precip x Temp)",
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

def generate_validation_plots():
    print("=========================================================")
    print("=== GENERATING DENGUE VALIDATION & ROC DIAGNOSTIC PLOTS ===")
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
    val_mask = (df_clean["year"] >= 2023) & (df_clean["year"] <= 2024)
    df_val = df_clean[val_mask].copy()
    
    # Load zone models and run inference on validation set
    models = {}
    for z in range(1, 7):
        m_path = f"dengue/models/dengue_model_zone_{z}.joblib"
        if os.path.exists(m_path):
            models[z] = joblib.load(m_path)
            
    df_val["pred_log"] = 0.0
    for z in range(1, 7):
        z_mask = df_val["climate_zone"] == float(z)
        if np.any(z_mask) and z in models:
            df_val.loc[z_mask, "pred_log"] = models[z].predict(df_val.loc[z_mask, DYNAMIC_FEATURES])
            
    df_val["pred_inc"] = np.clip(np.expm1(df_val["pred_log"]), 0, None)
    df_val["pred_cases"] = (df_val["pred_inc"] / 100000.0) * df_val["population"]
    
    out_dir = "final/outputs/graphs"
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. State-Level Actual vs Predicted Time-Series (2023-2024 Validation)
    # -------------------------------------------------------------
    state_weekly = df_val.groupby("date")[["cases", "pred_cases"]].sum().reset_index()
    
    st_r2 = r2_score(state_weekly["cases"].values, state_weekly["pred_cases"].values)
    st_pear, _ = pearsonr(state_weekly["cases"].values, state_weekly["pred_cases"].values)
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(state_weekly["date"], state_weekly["cases"] / 1e3, label="Actual Ground Truth", color="#1f77b4", lw=2.2)
    ax.plot(state_weekly["date"], state_weekly["pred_cases"] / 1e3, label="LightGBM Model Prediction", color="#ff7f0e", lw=2.2, linestyle="--")
    
    ax.text(0.04, 0.88, f"State-Level $R^2 = {st_r2:.4f}$\nPearson $r = {st_pear:.4f}$", transform=ax.transAxes, fontsize=13, fontfamily="serif", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    
    ax.grid(False)
    ax.set_xlabel("Validation Date (2023–2024 Holdout Set)", fontsize=14, fontweight="bold", fontstyle="italic", fontfamily="serif")
    ax.set_ylabel("Weekly Dengue Cases (in Thousands)", fontsize=14, fontweight="bold", fontstyle="italic", fontfamily="serif")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=13, loc="upper right", frameon=False)
    
    plt.tight_layout()
    fig.savefig(f"{out_dir}/dengue_validation_actual_vs_predicted.eps", format="eps")
    fig.savefig(f"{out_dir}/dengue_validation_actual_vs_predicted.png", dpi=600)
    plt.close()
    print(f"Saved dengue_validation_actual_vs_predicted.png & .eps (R2 = {st_r2:.4f}, r = {st_pear:.4f})")

    # -------------------------------------------------------------
    # 2. Outbreak Detection ROC Curve (Empirical AUC = 0.9349)
    # -------------------------------------------------------------
    act_inc = df_val["incidence_rate"].values
    pred_inc = df_val["pred_inc"].values
    threshold = np.percentile(act_inc, 75)
    y_true = (act_inc >= threshold).astype(int)
    y_score = pred_inc
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#d62728", lw=2.5, label=f"LightGBM Dengue Model (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Baseline (AUC = 0.50)")
    
    ax.grid(False)
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold", fontstyle="italic", fontfamily="serif")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold", fontstyle="italic", fontfamily="serif")
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11, loc="lower right", frameon=False)
    
    plt.tight_layout()
    fig.savefig(f"{out_dir}/dengue_outbreak_roc_curve.eps", format="eps")
    fig.savefig(f"{out_dir}/dengue_outbreak_roc_curve.png", dpi=600)
    plt.close()
    print(f"Saved dengue_outbreak_roc_curve.png & .eps (Empirical AUC = {roc_auc:.4f})")

    # -------------------------------------------------------------
    # 3. Actual vs Predicted Log-Log Scatter Plot (2023-2024 Validation)
    # -------------------------------------------------------------
    val_mae = mean_absolute_error(act_inc, pred_inc)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(act_inc, pred_inc, alpha=0.08, color="#2ca02c", edgecolors="none", s=15)
    
    # 1:1 Reference Line (y = x)
    max_val = max(act_inc.max(), pred_inc.max())
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=2.0, label="1:1 Perfect Fit")
    
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    
    ax.text(0.05, 0.90, f"State-Level $R^2 = {st_r2:.4f}$\nMAE = {val_mae:.2f} /100k", transform=ax.transAxes, fontsize=12, fontfamily="serif", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    
    ax.grid(False)
    ax.set_xlabel("Actual Incidence Rate (per 100k)", fontsize=13, fontweight="bold", fontstyle="italic", fontfamily="serif")
    ax.set_ylabel("Predicted Incidence Rate (per 100k)", fontsize=13, fontweight="bold", fontstyle="italic", fontfamily="serif")
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11, loc="lower right", frameon=False)
    
    plt.tight_layout()
    fig.savefig(f"{out_dir}/dengue_validation_scatter.eps", format="eps")
    fig.savefig(f"{out_dir}/dengue_validation_scatter.png", dpi=600)
    plt.close()
    print("Saved dengue_validation_scatter.png & .eps")

    # -------------------------------------------------------------
    # 4. LightGBM Feature Importance Plot
    # -------------------------------------------------------------
    importances = np.zeros(len(DYNAMIC_FEATURES))
    for z in range(1, 7):
        if z in models:
            importances += models[z].feature_importances_
    importances /= len(models)
    
    feat_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in DYNAMIC_FEATURES],
        "Importance": importances
    }).sort_values("Importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2ca02c", edgecolor="none", height=0.65)
    
    ax.grid(False)
    ax.set_xlabel("Mean Feature Importance (Split Count across Macro-Zones)", fontsize=13, fontweight="bold", fontstyle="italic", fontfamily="serif")
    ax.tick_params(axis="both", labelsize=11)
    
    plt.tight_layout()
    fig.savefig(f"{out_dir}/dengue_feature_importance.eps", format="eps")
    fig.savefig(f"{out_dir}/dengue_feature_importance.png", dpi=600)
    plt.close()
    print("Saved dengue_feature_importance.png & .eps")
    
    print("All validation & diagnostic plots generated successfully!")

if __name__ == "__main__":
    generate_validation_plots()
