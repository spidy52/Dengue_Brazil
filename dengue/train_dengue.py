import os
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, roc_auc_score

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

def downcast_dtypes(df):
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    return df

def generate_time_features(df):
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = (df["epiweek"] % 100).astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int16)
    df["sin_week"] = np.sin(2 * np.pi * df["week"] / 52.17857).astype(np.float32)
    df["cos_week"] = np.cos(2 * np.pi * df["week"] / 52.17857).astype(np.float32)
    return df

def generate_dengue_features(df):
    """Engineer dynamic features. Returns (df_modified, feature_list)."""
    # Lags
    df["lag_1_incidence"] = df.groupby("geocode")["incidence_rate"].shift(1).astype(np.float32)
    df["lag_2_incidence"] = df.groupby("geocode")["incidence_rate"].shift(2).astype(np.float32)
    df["lag_52_incidence"] = df.groupby("geocode")["incidence_rate"].shift(52).astype(np.float32)
    df["lag_104_incidence"] = df.groupby("geocode")["incidence_rate"].shift(104).astype(np.float32)
    df["diff_lag_1"] = df["lag_1_incidence"] - df["lag_2_incidence"]
    
    # Cumulative cases (immunity index) - shift(1) to avoid leakage
    df["rolling_cases_52w"] = df.groupby("geocode")["cases"].shift(1).rolling(window=52, min_periods=1).sum().reset_index(level=0, drop=True).astype(np.float32)
    df["cum_incidence_52w"] = (df["rolling_cases_52w"] / df["population"]) * 100000.0
    
    # Historical baselines (use ≤2022 to get a clean seasonal baseline
    # not biased by the extreme 2023-2024 outbreaks)
    train_mask = df["year"] <= 2022
    
    # Dengue baseline
    hist_mean_inc = df[train_mask].groupby(["geocode", "week"])["incidence_rate"].mean().reset_index().rename(columns={"incidence_rate": "hist_mean_incidence"})
    df = df.merge(hist_mean_inc, on=["geocode", "week"], how="left")
    overall_week_mean = df[train_mask].groupby("week")["incidence_rate"].mean().to_dict()
    df["hist_mean_incidence"] = df["hist_mean_incidence"].fillna(df["week"].map(overall_week_mean))
    df["hist_mean_incidence"] = df["hist_mean_incidence"].fillna(df["incidence_rate"].mean()).astype(np.float32)
    
    # Temp baseline & anomaly
    hist_temp = df[train_mask].groupby(["geocode", "week"])["temp_med"].mean().reset_index().rename(columns={"temp_med": "hist_mean_temp"})
    df = df.merge(hist_temp, on=["geocode", "week"], how="left")
    df["hist_mean_temp"] = df["hist_mean_temp"].fillna(df["temp_med"].mean()).astype(np.float32)
    df["temp_anomaly"] = df["temp_med"] - df["hist_mean_temp"]
    
    # Precip baseline & anomaly
    hist_precip = df[train_mask].groupby(["geocode", "week"])["precip_tot"].mean().reset_index().rename(columns={"precip_tot": "hist_mean_precip"})
    df = df.merge(hist_precip, on=["geocode", "week"], how="left")
    df["hist_mean_precip"] = df["hist_mean_precip"].fillna(df["precip_tot"].mean()).astype(np.float32)
    df["precip_anomaly"] = df["precip_tot"] - df["hist_mean_precip"]
    
    # Return modified df (merges create new dataframe, must be returned)
    return df, DYNAMIC_FEATURES

def train_dengue():
    csv_path = "final_brazil_dengue.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "final_brazil_dengue.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Could not find final_brazil_dengue.csv in root or data/ directory.")
        
    print("Loading historical data for Dengue Training...")
    cols_to_load = [
        "date", "year", "epiweek", "geocode", "cases", "incidence_rate",
        "temp_med", "precip_tot", "rel_humid_med", "population", "climate_zone"
    ]
    df = pd.read_csv(csv_path, usecols=cols_to_load)
    
    df = df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    df = downcast_dtypes(df)
    df = generate_time_features(df)
    
    print("Generating dengue features (Dynamic features)...")
    df, _ = generate_dengue_features(df)
    
    # Drop rows with NaNs
    df = df.dropna(subset=DYNAMIC_FEATURES + ["incidence_rate"]).reset_index(drop=True)
    
    # ----------------------------------------------------
    # Train separate LightGBM model for each Climate Zone
    # ----------------------------------------------------
    print("Training separate LightGBM Regressors for each Climate Zone...")
    
    unique_zones = sorted(df["climate_zone"].unique())
    
    val_preds_list = []
    val_y_list = []
    
    train_residuals_list = []
    train_weeks_list = []
    
    for zone in unique_zones:
        df_z = df[df["climate_zone"] == zone]
        
        X_train_z = df_z[df_z["year"] <= 2023][DYNAMIC_FEATURES]
        y_train_z = df_z[df_z["year"] <= 2023]["incidence_rate"]
        
        X_val_z = df_z[df_z["year"] == 2024][DYNAMIC_FEATURES]
        y_val_z = df_z[df_z["year"] == 2024]["incidence_rate"]
        
        y_train_fit_z = y_train_z - X_train_z["lag_1_incidence"]
        y_val_fit_z = y_val_z - X_val_z["lag_1_incidence"]
        
        print(f"Zone {int(zone)}: Train shape: {X_train_z.shape}, Val shape: {X_val_z.shape}")
        
        model = lgb.LGBMRegressor(
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=127,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        model.fit(
            X_train_z, y_train_fit_z,
            eval_set=[(X_train_z, y_train_fit_z), (X_val_z, y_val_fit_z)],
            eval_metric="l2"
        )
        
        # Save model
        model_path = f"dengue/models/dengue_model_zone_{int(zone)}.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved Zone {int(zone)} model to {model_path}")
        
        # Validation prediction
        if len(X_val_z) > 0:
            preds_diff_z = model.predict(X_val_z)
            preds_inc_z = X_val_z["lag_1_incidence"].values + 0.45 * preds_diff_z
            preds_inc_z = np.clip(preds_inc_z, 0, None)
            
            val_preds_list.extend(preds_inc_z)
            val_y_list.extend(y_val_z.values)
            
        # Residual computation on train set for stochastic noise calibration
        if len(X_train_z) > 0:
            preds_train_diff_z = model.predict(X_train_z)
            preds_train_inc_z = X_train_z["lag_1_incidence"].values + 0.45 * preds_train_diff_z
            preds_train_inc_z = np.clip(preds_train_inc_z, 0, None)
            
            residuals_z = y_train_z.values - preds_train_inc_z
            train_residuals_list.extend(residuals_z)
            train_weeks_list.extend(df_z[df_z["year"] <= 2023]["week"].values)
            
    # Combine residuals to save residual_info.joblib
    train_res_df = pd.DataFrame({
        "week": train_weeks_list,
        "residual": train_residuals_list
    })
    week_residual_std = train_res_df.groupby("week")["residual"].std().fillna(0).to_dict()
    global_residual_std = train_res_df["residual"].std()
    
    residual_info = {
        "week_std": week_residual_std,
        "global_std": global_residual_std
    }
    joblib.dump(residual_info, "dengue/models/residual_info.joblib")
    print(f"Saved combined residual info (global_std={global_residual_std:.2f})")
    
    y_val = np.array(val_y_list)
    preds = np.array(val_preds_list)
    
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"Overall Zone-wise Model Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        "Model": "Dengue_LightGBM_Dynamic_Zonewise",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }])
    metrics_df.to_csv("final/outputs/metrics/dengue_metrics.csv", index=False)
    
    # Average Feature Importance across all zones
    avg_importance = np.zeros(len(DYNAMIC_FEATURES))
    for zone in unique_zones:
        z_model = joblib.load(f"dengue/models/dengue_model_zone_{int(zone)}.joblib")
        avg_importance += z_model.feature_importances_
    avg_importance /= len(unique_zones)
    
    imp_df = pd.DataFrame({"Feature": DYNAMIC_FEATURES, "Importance": avg_importance}).sort_values("Importance", ascending=False)
    imp_df.to_csv("final/outputs/feature_importance/dengue_importance.csv", index=False)
    
    # Plot average feature importance
    plt.figure(figsize=(10, 8))
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="#2ca02c")
    plt.title("Average Feature Importance for Zone-wise Dengue Models", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    plt.savefig("final/outputs/feature_importance/dengue_importance.png", dpi=150)
    plt.close()
    
    # Learning curve of Zone 5 model
    model_z5 = joblib.load("dengue/models/dengue_model_zone_5.joblib")
    eval_results = model_z5.evals_result_
    train_loss = eval_results["training"]["l2"]
    val_loss = eval_results["valid_1"]["l2"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss", color="#1f77b4", lw=2)
    plt.plot(val_loss, label="Validation Loss", color="#ff7f0e", lw=2)
    plt.title("Dengue Zone 5 Model Learning Curve", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("final/outputs/graphs/dengue_learning_curve.png", dpi=150)
    plt.close()
    
    # Plot ROC Curve
    threshold = 100.0
    y_val_binary = (y_val > threshold).astype(int)
    if len(np.unique(y_val_binary)) > 1:
        fpr, tpr, roc_thresholds = roc_curve(y_val_binary, preds)
        auc_score = roc_auc_score(y_val_binary, preds)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color="#2ca02c", lw=3, label=f"ROC Curve (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(f"ROC Curve for Dengue Outbreak Detection (Threshold > {threshold} per 100k)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (TPR)", fontsize=12)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("final/outputs/graphs/dengue_roc_curve.png", dpi=150)
        plt.close()
        print(f"Saved ROC curve plot (AUC={auc_score:.4f})")
        
    print("Dengue training finished successfully.")
    
if __name__ == "__main__":
    train_dengue()
