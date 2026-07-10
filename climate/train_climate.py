import os
import gc
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

def create_dirs():
    dirs = [
        "climate/models",
        "climate/outputs",
        "final/outputs/csv",
        "final/outputs/graphs",
        "final/outputs/maps",
        "final/outputs/metrics",
        "final/outputs/feature_importance"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def downcast_dtypes(df):
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    return df

def generate_time_features(df):
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])
    # Extract week from epiweek (epiweek % 100) or date.
    # Since we verified epiweek % 100 is the official week, let's use that
    df["week"] = (df["epiweek"] % 100).astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int16)
    
    # Trigonometric encoding for seasonality
    df["sin_week"] = np.sin(2 * np.pi * df["week"] / 52.17857).astype(np.float32)
    df["cos_week"] = np.cos(2 * np.pi * df["week"] / 52.17857).astype(np.float32)
    return df

def generate_features_for_target(df, target):
    features = []
    
    # 1. Lags
    lags = [1, 2, 3, 4, 8, 12, 16, 20, 26, 39, 52]
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby("geocode")[target].shift(lag).astype(np.float32)
        features.append(col_name)
        
    # 2. Rolling features (using shift(1) to avoid leaking current target value)
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
        
    # 3. Time features
    time_cols = ["week", "month", "sin_week", "cos_week"]
    features.extend(time_cols)
    
    return features

def train_climate():
    create_dirs()
    
    csv_path = "final_brazil_dengue.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "final_brazil_dengue.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find final_brazil_dengue.csv in root or data/ directory.")
        
    targets = [
        "temp_min", "temp_med", "temp_max",
        "precip_med", "precip_tot", "pressure_med",
        "rel_humid_med", "thermal_range", "rainy_days"
    ]
    
    metrics_list = []
    
    print("Starting Climate Models Training...")
    
    for target in targets:
        print(f"\n--- Training model for target: {target} ---")
        
        # Load only necessary columns to keep memory low
        print("Loading data...")
        df = pd.read_csv(csv_path, usecols=["date", "year", "epiweek", "geocode", target])
        
        # Deduplicate
        df = df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
        df = downcast_dtypes(df)
        df = generate_time_features(df)
        
        # Generate target-specific features
        print("Generating features...")
        features = generate_features_for_target(df, target)
        
        # Calculate historical weekly mean of target for geocode-week using training set (year <= 2021)
        print("Computing historical weekly baseline...")
        train_mask = df["year"] <= 2021
        hist_mean_df = df[train_mask].groupby(["geocode", "week"])[target].mean().reset_index().rename(columns={target: f"hist_mean_{target}"})
        
        # Merge back
        df = df.merge(hist_mean_df, on=["geocode", "week"], how="left")
        
        # Fill missing values
        overall_week_mean = df[train_mask].groupby("week")[target].mean().to_dict()
        df[f"hist_mean_{target}"] = df[f"hist_mean_{target}"].fillna(df["week"].map(overall_week_mean))
        df[f"hist_mean_{target}"] = df[f"hist_mean_{target}"].fillna(df[target].mean()).astype(np.float32)
        
        # Append hist_mean to features
        features.append(f"hist_mean_{target}")
        
        # Drop rows with NaNs in features
        df = df.dropna(subset=features).reset_index(drop=True)
        
        # Train / Validation Split
        # Training: 2010–2021
        # Validation: 2022–2024
        X_train = df[df["year"] <= 2021][features]
        y_train = df[df["year"] <= 2021][target]
        
        X_val = df[(df["year"] >= 2022) & (df["year"] <= 2024)][features]
        y_val = df[(df["year"] >= 2022) & (df["year"] <= 2024)][target]
        
        print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")
        
        # Train LightGBM model
        print("Fitting LightGBM model...")
        model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        model.fit(X_train, y_train)
        
        # Save model
        model_path = f"climate/models/climate_{target}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")
        
        # Validation Evaluation
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        print(f"Validation Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        metrics_list.append({
            "Target": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        
        # Save feature importance CSV
        importance = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values("Importance", ascending=False)
        imp_path = f"final/outputs/feature_importance/climate_{target}_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        
        # Plot and save feature importance
        plt.figure(figsize=(10, 6))
        # Style with professional dark aesthetics or neat whitegrid
        plt.style.use("seaborn-v0_8-whitegrid")
        top_n = imp_df.head(15)
        plt.barh(top_n["Feature"][::-1], top_n["Importance"][::-1], color="#1f77b4")
        plt.title(f"Feature Importance for {target}", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Importance score", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"final/outputs/feature_importance/climate_{target}_importance.png", dpi=150)
        plt.close()
        
        # Cleanup
        del df, X_train, y_train, X_val, y_val, model, preds
        gc.collect()
        
    # Save overall metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("final/outputs/metrics/climate_metrics.csv", index=False)
    print("\nClimate models training finished successfully. Metrics saved to final/outputs/metrics/climate_metrics.csv")

if __name__ == "__main__":
    train_climate()
