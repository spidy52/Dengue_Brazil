import os
import gc
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

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
    """Engineer high-precision dynamic epidemiological features."""
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
    
    # Train mask for baseline computation (up to 2022)
    train_mask = df["year"] <= 2022
    
    muni_stats = df[train_mask].groupby("geocode")["incidence_rate"].agg(["mean", "max"]).reset_index().rename(columns={"mean": "muni_mean_inc", "max": "muni_max_inc"})
    df = df.merge(muni_stats, on="geocode", how="left")
    df["muni_mean_inc"] = df["muni_mean_inc"].fillna(df["incidence_rate"].mean()).astype(np.float32)
    df["muni_max_inc"] = df["muni_max_inc"].fillna(df["incidence_rate"].max()).astype(np.float32)
    
    df["log_inc"] = np.log1p(df["incidence_rate"]).astype(np.float32)
    df["log_lag1"] = np.log1p(df["lag_1_inc"]).astype(np.float32)
    
    return df, DYNAMIC_FEATURES

def train_dengue():
    os.makedirs("dengue/models", exist_ok=True)
    os.makedirs("final/outputs/csv", exist_ok=True)
    os.makedirs("final/outputs/metrics", exist_ok=True)
    
    csv_path = "final_brazil_dengue.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "final_brazil_dengue.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Could not find final_brazil_dengue.csv in root or data/ directory.")
        
    print("Loading historical data for High-Precision Dengue Model Training...")
    cols_to_load = [
        "date", "year", "epiweek", "geocode", "uf", "cases", "incidence_rate",
        "temp_med", "precip_tot", "rel_humid_med", "population", "climate_zone"
    ]
    df = pd.read_csv(csv_path, usecols=cols_to_load)
    
    df = df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    df = downcast_dtypes(df)
    df = generate_time_features(df)
    
    print("Engineering dynamic features...")
    df, _ = generate_dengue_features(df)
    df = df.dropna(subset=DYNAMIC_FEATURES + ["incidence_rate"]).reset_index(drop=True)
    
    unique_zones = sorted(df["climate_zone"].unique())
    
    val_df_list = []
    
    print("\nTraining High-Precision Zone-wise LightGBM Regressors...")
    for zone in unique_zones:
        df_z = df[df["climate_zone"] == zone]
        
        train_mask = df_z["year"] <= 2022
        val_mask = (df_z["year"] >= 2023) & (df_z["year"] <= 2024)
        
        X_train_z = df_z[train_mask][DYNAMIC_FEATURES]
        y_train_z_log = df_z[train_mask]["log_inc"]
        
        X_val_z = df_z[val_mask][DYNAMIC_FEATURES]
        y_val_z_act = df_z[val_mask]["incidence_rate"]
        
        print(f"Zone {int(zone)}: Train size = {X_train_z.shape[0]:,}, Val size = {X_val_z.shape[0]:,}")
        
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=255,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        model.fit(X_train_z, y_train_z_log)
        
        model_path = f"dengue/models/dengue_model_zone_{int(zone)}.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved model to {model_path}")
        
        # Evaluate on validation set
        preds_log_val = model.predict(X_val_z)
        preds_inc_val = np.clip(np.expm1(preds_log_val), 0, None)
        
        df_val_sub = df_z[val_mask][["date", "geocode", "uf", "climate_zone", "week", "population", "cases", "incidence_rate", "log_inc"]].copy()
        df_val_sub["pred_log"] = preds_log_val
        df_val_sub["pred_inc"] = preds_inc_val
        df_val_sub["pred_cases"] = (df_val_sub["pred_inc"] / 100000.0) * df_val_sub["population"]
        val_df_list.append(df_val_sub)
        
        z_mae = mean_absolute_error(y_val_z_act.values, preds_inc_val)
        z_r2 = r2_score(y_val_z_act.values, preds_inc_val)
        print(f"  Zone {int(zone)} Val MAE: {z_mae:.2f}, Municipality R2: {z_r2:.4f}")
        
    df_val_all = pd.concat(val_df_list, ignore_index=True)
    
    # 1. Municipality-level dynamic evaluation (Weekly Cases & Log-Incidence)
    muni_cases_true = df_val_all["cases"].values
    muni_cases_pred = df_val_all["pred_cases"].values
    muni_cases_r2 = r2_score(muni_cases_true, muni_cases_pred)
    
    muni_log_true = df_val_all["log_inc"].values
    muni_log_pred = df_val_all["pred_log"].values
    muni_log_r2 = r2_score(muni_log_true, muni_log_pred)
    
    muni_inc_true = df_val_all["incidence_rate"].values
    muni_inc_pred = df_val_all["pred_inc"].values
    muni_mae = mean_absolute_error(muni_inc_true, muni_inc_pred)
    muni_pear_r, _ = pearsonr(muni_inc_true, muni_inc_pred)
    muni_spear_rho, _ = spearmanr(muni_inc_true, muni_inc_pred)
    
    # Outbreak ROC-AUC
    thresh_75 = np.percentile(muni_inc_true, 75)
    binary_true = (muni_inc_true > thresh_75).astype(int)
    roc_auc = roc_auc_score(binary_true, muni_inc_pred)
    
    # 2. State-level weekly cases dynamic evaluation
    state_weekly = df_val_all.groupby(["uf", "date"])[["cases", "pred_cases"]].sum().reset_index()
    state_r2 = r2_score(state_weekly["cases"].values, state_weekly["pred_cases"].values)
    state_mae = mean_absolute_error(state_weekly["cases"].values, state_weekly["pred_cases"].values)
    state_pear_r, _ = pearsonr(state_weekly["cases"].values, state_weekly["pred_cases"].values)
    state_spear_rho, _ = spearmanr(state_weekly["cases"].values, state_weekly["pred_cases"].values)
    
    # 3. Zone-level weekly incidence dynamic evaluation & residual std
    zone_weekly = df_val_all.groupby(["climate_zone", "week", "date"])[["cases", "pred_cases"]].sum().reset_index()
    zone_weekly["pop"] = df_val_all.groupby(["climate_zone", "week", "date"])["population"].sum().values
    
    zone_weekly["true_inc"] = (zone_weekly["cases"] / zone_weekly["pop"]) * 100000.0
    zone_weekly["pred_inc"] = (zone_weekly["pred_cases"] / zone_weekly["pop"]) * 100000.0
    zone_weekly["residual"] = zone_weekly["true_inc"] - zone_weekly["pred_inc"]
    
    zone_r2 = r2_score(zone_weekly["true_inc"].values, zone_weekly["pred_inc"].values)
    
    zone_residual_std = {}
    for z in range(1, 7):
        z_df = zone_weekly[zone_weekly["climate_zone"] == float(z)]
        w_std = z_df.groupby("week")["residual"].std().fillna(0.0).to_dict()
        g_std = float(z_df["residual"].std())
        zone_residual_std[z] = {"week_std": w_std, "global_std": max(g_std, 1.5)}
        
    joblib.dump(zone_residual_std, "dengue/models/residual_info.joblib")
    
    state_rmse = np.sqrt(mean_squared_error(state_weekly["cases"].values, state_weekly["pred_cases"].values))
    zone_mae = mean_absolute_error(zone_weekly["true_inc"].values, zone_weekly["pred_inc"].values)
    zone_rmse = np.sqrt(mean_squared_error(zone_weekly["true_inc"].values, zone_weekly["pred_inc"].values))
    muni_rmse = np.sqrt(mean_squared_error(muni_inc_true, muni_inc_pred))
    
    print("\n=========================================================")
    print("=== DYNAMICALLY COMPUTED MODEL VALIDATION METRICS ===")
    print("=========================================================")
    print(f"  State-Level Weekly Cases R2 Score : {state_r2:.4f} | MAE: {state_mae:.2f} cases | RMSE: {state_rmse:.2f} cases")
    print(f"  Zone-Level Weekly Incidence R2    : {zone_r2:.4f} | MAE: {zone_mae:.2f} /100k | RMSE: {zone_rmse:.2f} /100k")
    print(f"  Municipality-Level Weekly Cases R2: {muni_cases_r2:.4f} | MAE: {muni_mae:.2f} /100k | RMSE: {muni_rmse:.2f} /100k")
    print(f"  Municipality-Level Log-Incidence R2: {muni_log_r2:.4f}")
    print(f"  State-Level Pearson R             : {state_pear_r:.4f}")
    print(f"  State-Level Spearman Rho          : {state_spear_rho:.4f}")
    print(f"  Outbreak Detection ROC-AUC       : {roc_auc:.4f}")
    print("=========================================================")
    
    metrics_df = pd.DataFrame([
        {
            "Model": "LightGBM_Dynamic_Zonewise_HighPrecision",
            "Level": "State_Level_Weekly_Cases",
            "R2": round(float(state_r2), 4),
            "Pearson_R": round(float(state_pear_r), 4),
            "Spearman_Rho": round(float(state_spear_rho), 4),
            "MAE": f"{round(float(state_mae), 2)} cases",
            "RMSE": f"{round(float(state_rmse), 2)} cases",
            "Outbreak_ROC_AUC": round(float(roc_auc), 4)
        },
        {
            "Model": "LightGBM_Dynamic_Zonewise_HighPrecision",
            "Level": "Zone_Level_Weekly_Incidence",
            "R2": round(float(zone_r2), 4),
            "Pearson_R": round(float(state_pear_r), 4),
            "Spearman_Rho": round(float(state_spear_rho), 4),
            "MAE": f"{round(float(zone_mae), 2)} /100k",
            "RMSE": f"{round(float(zone_rmse), 2)} /100k",
            "Outbreak_ROC_AUC": round(float(roc_auc), 4)
        },
        {
            "Model": "LightGBM_Dynamic_Zonewise_HighPrecision",
            "Level": "Municipality_Level_Weekly_Cases",
            "R2": round(float(muni_cases_r2), 4),
            "Pearson_R": round(float(muni_pear_r), 4),
            "Spearman_Rho": round(float(muni_spear_rho), 4),
            "MAE": f"{round(float(muni_mae), 2)} /100k",
            "RMSE": f"{round(float(muni_rmse), 2)} /100k",
            "Outbreak_ROC_AUC": round(float(roc_auc), 4)
        },
        {
            "Model": "LightGBM_Dynamic_Zonewise_HighPrecision",
            "Level": "Municipality_Level_Log_Incidence",
            "R2": round(float(muni_log_r2), 4),
            "Pearson_R": round(float(muni_pear_r), 4),
            "Spearman_Rho": round(float(muni_spear_rho), 4),
            "MAE": f"{round(float(muni_mae), 2)} /100k",
            "RMSE": f"{round(float(muni_rmse), 2)} /100k",
            "Outbreak_ROC_AUC": round(float(roc_auc), 4)
        }
    ])
    metrics_df.to_csv("final/outputs/metrics/dengue_metrics.csv", index=False)
    print("Saved dynamically evaluated high-precision metrics to final/outputs/metrics/dengue_metrics.csv")

if __name__ == "__main__":
    train_dengue()
