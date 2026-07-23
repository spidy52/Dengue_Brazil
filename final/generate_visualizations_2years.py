import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def generate_visualizations_2years():
    print("Generating 2-Year Zone-wise & Overall Improved Model Visualizations...")
    os.makedirs("final/outputs_2years/graphs", exist_ok=True)
    
    hist_csv = "final_brazil_dengue.csv"
    if not os.path.exists(hist_csv):
        hist_csv = os.path.join("data", "final_brazil_dengue.csv")
    zone_csv = "final/outputs_2years/csv/zone_dengue_2025_2026.csv"

    hist_df = pd.read_csv(hist_csv, usecols=["date", "year", "cases", "population", "climate_zone", "geocode"])
    hist_df = hist_df.drop_duplicates(subset=["date", "geocode"]).sort_values(["geocode", "date"]).reset_index(drop=True)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
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

    print("Aggregating historical zone data...")
    hist_zone_df = hist_recent.groupby("date", group_keys=False).apply(agg_hist_zones).reset_index(drop=True)

    if os.path.exists(zone_csv):
        zone_fore = pd.read_csv(zone_csv)
        zone_fore["date"] = pd.to_datetime(zone_fore["date"])

        for zone in sorted(zone_fore["climate_zone"].unique()):
            hz = hist_zone_df[hist_zone_df["climate_zone"] == zone].sort_values("date")
            fz = zone_fore[zone_fore["climate_zone"] == zone].sort_values("date")

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(hz["date"], hz["incidence_rate"], label="Historical (2020-2024)", color="#1f77b4", lw=2.5)
            ax.plot(fz["date"], fz["incidence_rate"], label="Improved 2-Year Forecast (2025-2026)", color="#2ca02c", lw=2.5, linestyle="--")

            fstd = fz["incidence_rate"].rolling(4, center=True, min_periods=1).std().fillna(0)
            ax.fill_between(fz["date"], (fz["incidence_rate"] - fstd).clip(lower=0), fz["incidence_rate"] + fstd, color="#2ca02c", alpha=0.18, label="Forecast ±1σ")

            forecast_start = fz["date"].min()
            ax.axvline(x=forecast_start, color="gray", linestyle=":", lw=1.5, alpha=0.7)
            ax.text(forecast_start, ax.get_ylim()[1] * 0.92, "  2-Year Forecast Horizon →", fontsize=10, color="gray", va="top")

            ax.set_title(f"Improved 2-Year Dengue Forecast - Climate Zone {int(zone)} (History vs 2025-2026 Forecast)", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Incidence Rate (per 100k)", fontsize=12)
            ax.legend(loc="upper left", fontsize=10)
            plt.tight_layout()
            
            out_file = f"final/outputs_2years/graphs/dengue_forecast_improved_2years_zone_{int(zone)}.png"
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"  Saved 2-Year Zone {int(zone)} plot to {out_file}")

        # Combined Multi-Zone 2-Year Graph
        plt.figure(figsize=(14, 7))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, zone in enumerate(sorted(zone_fore["climate_zone"].unique())):
            fz = zone_fore[zone_fore["climate_zone"] == zone].sort_values("date")
            c = colors[i % len(colors)]
            plt.plot(fz["date"], fz["incidence_rate"], label=f"Zone {int(zone)}", lw=2, color=c)
            fstd = fz["incidence_rate"].rolling(4, center=True, min_periods=1).std().fillna(0)
            plt.fill_between(fz["date"], (fz["incidence_rate"] - fstd).clip(0), fz["incidence_rate"] + fstd, alpha=0.10, color=c)

        plt.title("Improved 2-Year Dengue Incidence Forecast by Climate Zone (2025–2026)", fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Incidence Rate (per 100k)", fontsize=12)
        plt.legend(title="Climate Zone", fontsize=10)
        plt.tight_layout()
        plt.savefig("final/outputs_2years/graphs/dengue_forecast_improved_2years_combined_zones.png", dpi=150)
        plt.close()

    print("All 2-Year Zone-wise plots saved successfully!")

if __name__ == "__main__":
    generate_visualizations_2years()
