import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.linewidth": 2.2,
    "axes.edgecolor": "#000000",
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size": 6,
    "ytick.major.size": 6
})

def generate_visualizations():
    print("=========================================================")
    print("=== GENERATING 5-YEAR DENGUE VISUALIZATIONS (2025-2030) ===")
    print("=========================================================")
    
    csv_path = "final_brazil_dengue.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "final_brazil_dengue.csv")
        
    cols_to_load = ["date", "year", "epiweek", "geocode", "uf", "cases", "population", "climate_zone"]
    df_hist = pd.read_csv(csv_path, usecols=cols_to_load)
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    
    muni_pop = df_hist[df_hist["year"] == 2024].drop_duplicates("geocode")
    zone_pop_exact = muni_pop.groupby("climate_zone")["population"].sum().to_dict()
    
    zone_weekly = df_hist.groupby(["climate_zone", "date"])["cases"].sum().reset_index()
    zone_weekly["zone_pop"] = zone_weekly["climate_zone"].map(zone_pop_exact)
    zone_weekly["true_incidence_rate"] = (zone_weekly["cases"] / zone_weekly["zone_pop"]) * 100000.0
    
    forecast_csv = "final/outputs/csv/zone_dengue_2025_2030.csv"
    if not os.path.exists(forecast_csv):
        print("Running dengue forecast pipeline first...")
        from dengue.predict_dengue import run_dengue_prediction
        run_dengue_prediction()
        
    df_forecast = pd.read_csv(forecast_csv)
    df_forecast["date"] = pd.to_datetime(df_forecast["date"])
    
    out_dir_5yr = "final/outputs/graphs"
    os.makedirs(out_dir_5yr, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    for zone in range(1, 7):
        hz = zone_weekly[zone_weekly["climate_zone"] == float(zone)].sort_values("date").copy()
        hz_actual = hz[(hz["date"].dt.year >= 2018) & (hz["date"] <= pd.to_datetime("2024-06-02"))].copy()
        
        last_dt = hz_actual["date"].iloc[-1]
        last_inc = hz_actual["true_incidence_rate"].iloc[-1]
        
        z_fore = df_forecast[df_forecast["climate_zone"] == float(zone)].sort_values("date").copy()
        
        df_val_sub = z_fore[z_fore["date"] <= pd.to_datetime("2025-12-28")].copy()
        anchor_val = pd.DataFrame([{"date": last_dt, "incidence_rate": last_inc, "sigma": 0.0}])
        df_val_2025 = pd.concat([anchor_val, df_val_sub], ignore_index=True)
        
        last_buf_dt = df_val_sub["date"].iloc[-1]
        last_buf_inc = df_val_sub["incidence_rate"].iloc[-1]
        df_fut_sub = z_fore[z_fore["date"] >= pd.to_datetime("2026-01-04")].copy()
        anchor_fut = pd.DataFrame([{"date": last_buf_dt, "incidence_rate": last_buf_inc, "sigma": df_fut_sub["sigma"].iloc[0]}])
        df_fut = pd.concat([anchor_fut, df_fut_sub], ignore_index=True)
        
        fig, ax = plt.subplots(figsize=(12, 5.2), facecolor="white")
        
        ax.plot(hz_actual["date"], hz_actual["true_incidence_rate"], label="Historical", color="#1f77b4", lw=2.0)
        ax.plot(df_val_2025["date"], df_val_2025["incidence_rate"], label="Validation Buffer", color="#2ca02c", lw=2.0, linestyle="--")
        ax.plot(df_fut["date"], df_fut["incidence_rate"], label="Forecast", color="#ff7f0e", lw=2.0, linestyle=":")
        
        lower_fut = (df_fut["incidence_rate"] - df_fut["sigma"]).clip(0)
        upper_fut = df_fut["incidence_rate"] + df_fut["sigma"]
        
        ax.plot(df_fut["date"], lower_fut, color="#ff7f0e", lw=1.0, linestyle="--")
        ax.plot(df_fut["date"], upper_fut, color="#ff7f0e", lw=1.0, linestyle="--")
        
        ax.axvline(x=pd.to_datetime("2024-06-02"), color="gray", linestyle=":", lw=1.2)
        ax.axvline(x=pd.to_datetime("2025-12-31"), color="red", linestyle="--", lw=1.2)
        
        y_max = max(hz_actual["true_incidence_rate"].max(), df_val_2025["incidence_rate"].max(), df_fut["incidence_rate"].max()) * 1.05
        ax.text(pd.to_datetime("2024-06-10"), y_max * 0.92, "Forecast ->", color="gray", fontsize=11, fontfamily="serif")
        
        ax.grid(False)
        ax.set_xlabel("Date", fontsize=14, fontweight="bold", fontstyle="italic", fontfamily="serif")
        ax.set_ylabel("Incidence Rate (per 100k)", fontsize=14, fontweight="bold", fontstyle="italic", fontfamily="serif")
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=12, loc="upper right", frameon=False)
        
        plt.tight_layout()
        
        # Save EPS vector format (clean vector bounds, NO PostScript alpha fill artifacts)
        fig.savefig(f"{out_dir_5yr}/dengue_forecast_zone_{zone}.eps", format="eps", bbox_inches="tight")
        fig.savefig(f"figures/dengue_forecast_zone_{zone}_5years.eps", format="eps", bbox_inches="tight")
        
        # Add smooth fill for PNG and PDF exports
        ax.fill_between(df_fut["date"], lower_fut, upper_fut, color="#ff7f0e", alpha=0.25, label="Forecast ±1σ Band")
        ax.legend(fontsize=12, loc="upper right", frameon=False)
        
        fig.savefig(f"{out_dir_5yr}/dengue_forecast_zone_{zone}.png", dpi=600, bbox_inches="tight")
        fig.savefig(f"figures/dengue_forecast_zone_{zone}_5years.png", dpi=600, bbox_inches="tight")
        fig.savefig(f"{out_dir_5yr}/dengue_forecast_zone_{zone}.pdf", bbox_inches="tight")
        plt.close()

    # 6-Panel Combined 5-Year Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), facecolor="white", sharex=True)
    axes = axes.flatten()
    zone_names = {
        1: "Zone 1 (Equatorial Amazon)", 2: "Zone 2 (Cerrado North)",
        3: "Zone 3 (Semi-Arid NE)", 4: "Zone 4 (Central-West)",
        5: "Zone 5 (Southeast Core)", 6: "Zone 6 (Southern Temperate)"
    }
    
    for i, z in enumerate(range(1, 7)):
        ax = axes[i]
        hz = zone_weekly[zone_weekly["climate_zone"] == float(z)].sort_values("date").copy()
        hz_actual = hz[(hz["date"].dt.year >= 2018) & (hz["date"] <= pd.to_datetime("2024-06-02"))].copy()
        last_dt = hz_actual["date"].iloc[-1]
        last_inc = hz_actual["true_incidence_rate"].iloc[-1]
        z_fore = df_forecast[df_forecast["climate_zone"] == float(z)].sort_values("date").copy()
        df_val_sub = z_fore[z_fore["date"] <= pd.to_datetime("2025-12-28")].copy()
        anchor_val = pd.DataFrame([{"date": last_dt, "incidence_rate": last_inc, "sigma": 0.0}])
        df_val_2025 = pd.concat([anchor_val, df_val_sub], ignore_index=True)
        last_buf_dt = df_val_sub["date"].iloc[-1]
        last_buf_inc = df_val_sub["incidence_rate"].iloc[-1]
        df_fut_sub = z_fore[z_fore["date"] >= pd.to_datetime("2026-01-04")].copy()
        anchor_fut = pd.DataFrame([{"date": last_buf_dt, "incidence_rate": last_buf_inc, "sigma": df_fut_sub["sigma"].iloc[0]}])
        df_fut = pd.concat([anchor_fut, df_fut_sub], ignore_index=True)

        ax.plot(hz_actual["date"], hz_actual["true_incidence_rate"], label="Historical", color="#1f77b4", lw=1.8)
        ax.plot(df_val_2025["date"], df_val_2025["incidence_rate"], label="Validation Buffer", color="#2ca02c", lw=1.8, linestyle="--")
        ax.plot(df_fut["date"], df_fut["incidence_rate"], label="Forecast", color="#ff7f0e", lw=1.8, linestyle=":")
        
        lower_fut = (df_fut["incidence_rate"] - df_fut["sigma"]).clip(0)
        upper_fut = df_fut["incidence_rate"] + df_fut["sigma"]
        
        ax.plot(df_fut["date"], lower_fut, color="#ff7f0e", lw=0.8, linestyle="--")
        ax.plot(df_fut["date"], upper_fut, color="#ff7f0e", lw=0.8, linestyle="--")
        
        ax.axvline(x=pd.to_datetime("2024-06-02"), color="gray", linestyle=":", lw=1.0)
        ax.axvline(x=pd.to_datetime("2025-12-31"), color="red", linestyle="--", lw=1.0)
        ax.grid(False)
        ax.text(0.03, 0.90, zone_names[z], transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#333333")
        ax.set_ylabel("Incidence Rate (per 100k)", fontsize=8.5, color="#333333")
        if i >= 4:
            ax.set_xlabel("Date", fontsize=9, color="#333333")
        ax.tick_params(colors="#333333", labelsize=8.5)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    plt.tight_layout()
    # Save EPS vector without fill_between
    plt.savefig("figures/dengue_forecast_combined_zones_5years.eps", format="eps", bbox_inches="tight")
    plt.savefig(f"{out_dir_5yr}/dengue_forecast_combined_zones_5years.eps", format="eps", bbox_inches="tight")

    # Add fill_between for PNG and PDF exports
    for i, z in enumerate(range(1, 7)):
        ax = axes[i]
        z_fore = df_forecast[df_forecast["climate_zone"] == float(z)].sort_values("date").copy()
        df_val_sub = z_fore[z_fore["date"] <= pd.to_datetime("2025-12-28")].copy()
        df_fut_sub = z_fore[z_fore["date"] >= pd.to_datetime("2026-01-04")].copy()
        last_buf_dt = df_val_sub["date"].iloc[-1]
        last_buf_inc = df_val_sub["incidence_rate"].iloc[-1]
        anchor_fut = pd.DataFrame([{"date": last_buf_dt, "incidence_rate": last_buf_inc, "sigma": df_fut_sub["sigma"].iloc[0]}])
        df_fut = pd.concat([anchor_fut, df_fut_sub], ignore_index=True)
        lower_fut = (df_fut["incidence_rate"] - df_fut["sigma"]).clip(0)
        upper_fut = df_fut["incidence_rate"] + df_fut["sigma"]
        ax.fill_between(df_fut["date"], lower_fut, upper_fut, color="#ff7f0e", alpha=0.25, label="Forecast ±1σ Band")
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    plt.savefig("figures/dengue_forecast_combined_zones_5years.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{out_dir_5yr}/dengue_forecast_combined_zones_5years.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{out_dir_5yr}/dengue_forecast_combined_zones_5years.pdf", bbox_inches="tight")
    plt.close()

    print("5-Year Visualizations generated successfully!")

if __name__ == "__main__":
    generate_visualizations()
