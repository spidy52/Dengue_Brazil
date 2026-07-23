import os
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    """Engineer dynamic features."""
    df["lag_1_incidence"] = df.groupby("geocode")["incidence_rate"].shift(1).astype(np.float32)
    df["lag_2_incidence"] = df.groupby("geocode")["incidence_rate"].shift(2).astype(np.float32)
    df["lag_52_incidence"] = df.groupby("geocode")["incidence_rate"].shift(52).astype(np.float32)
    df["lag_104_incidence"] = df.groupby("geocode")["incidence_rate"].shift(104).astype(np.float32)
    df["diff_lag_1"] = df["lag_1_incidence"] - df["lag_2_incidence"]
    
    df["rolling_cases_52w"] = df.groupby("geocode")["cases"].shift(1).rolling(window=52, min_periods=1).sum().reset_index(level=0, drop=True).astype(np.float32)
    df["cum_incidence_52w"] = (df["rolling_cases_52w"] / df["population"]) * 100000.0
    
    # Historical baseline up to 2023 for baseline comparison
    train_mask = df["year"] <= 2023
    
    hist_mean_inc = df[train_mask].groupby(["geocode", "week"])["incidence_rate"].mean().reset_index().rename(columns={"incidence_rate": "hist_mean_incidence"})
    df = df.merge(hist_mean_inc, on=["geocode", "week"], how="left")
    overall_week_mean = df[train_mask].groupby("week")["incidence_rate"].mean().to_dict()
    df["hist_mean_incidence"] = df["hist_mean_incidence"].fillna(df["week"].map(overall_week_mean))
    df["hist_mean_incidence"] = df["hist_mean_incidence"].fillna(df["incidence_rate"].mean()).astype(np.float32)
    
    hist_temp = df[train_mask].groupby(["geocode", "week"])["temp_med"].mean().reset_index().rename(columns={"temp_med": "hist_mean_temp"})
    df = df.merge(hist_temp, on=["geocode", "week"], how="left")
    df["hist_mean_temp"] = df["hist_mean_temp"].fillna(df["temp_med"].mean()).astype(np.float32)
    df["temp_anomaly"] = df["temp_med"] - df["hist_mean_temp"]
    
    hist_precip = df[train_mask].groupby(["geocode", "week"])["precip_tot"].mean().reset_index().rename(columns={"precip_tot": "hist_mean_precip"})
    df = df.merge(hist_precip, on=["geocode", "week"], how="left")
    df["hist_mean_precip"] = df["hist_mean_precip"].fillna(df["precip_tot"].mean()).astype(np.float32)
    df["precip_anomaly"] = df["precip_tot"] - df["hist_mean_precip"]
    
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
        
    print("Loading historical data for Improved Dengue Training (up to 2024)...")
    cols_to_load = [
        "date", "year", "epiweek", "geocode", "cases", "incidence_rate",
        "temp_med", "precip_tot", "rel_humid_med", "population", "climate_zone"
    ]
    df = pd.read_csv(csv_path, usecols=cols_to_load)
    
    df = df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    df = downcast_dtypes(df)
    df = generate_time_features(df)
    
    print("Generating dynamic features...")
    df, _ = generate_dengue_features(df)
    df = df.dropna(subset=DYNAMIC_FEATURES + ["incidence_rate"]).reset_index(drop=True)
    
    print("Training improved Zone-wise LightGBM Regressors including 2024 epidemic data...")
    unique_zones = sorted(df["climate_zone"].unique())
    
    train_residuals_list = []
    train_weeks_list = []
    
    for zone in unique_zones:
        df_z = df[df["climate_zone"] == zone]
        
        # Train on ALL data up to 2024 so trees see high epidemic peaks!
        X_train_z = df_z[df_z["year"] <= 2024][DYNAMIC_FEATURES]
        y_train_z = df_z[df_z["year"] <= 2024]["incidence_rate"]
        
        y_train_fit_z = y_train_z - X_train_z["lag_1_incidence"]
        
        print(f"Zone {int(zone)}: Train shape: {X_train_z.shape}")
        
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=255,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            subsample=0.85,
            colsample_bytree=0.85
        )
        
        model.fit(X_train_z, y_train_fit_z)
        
        model_path = f"dengue/models/dengue_model_zone_{int(zone)}.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved Improved Zone {int(zone)} model to {model_path}")
        
        preds_train_diff_z = model.predict(X_train_z)
        preds_train_inc_z = X_train_z["lag_1_incidence"].values + 0.45 * preds_train_diff_z
        preds_train_inc_z = np.clip(preds_train_inc_z, 0, None)
        
        residuals_z = y_train_z.values - preds_train_inc_z
        train_residuals_list.extend(residuals_z)
        train_weeks_list.extend(df_z[df_z["year"] <= 2024]["week"].values)
            
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
    print(f"Saved improved residual info (global_std={global_residual_std:.2f})")
    print("Improved Dengue model training finished successfully.")
    
if __name__ == "__main__":
    train_dengue()
