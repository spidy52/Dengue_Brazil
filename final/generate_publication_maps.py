import os
import ssl
import json
import urllib.request
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

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

def download_brazil_geojson():
    local_path = "data/brazil_states.geojson"
    if os.path.exists(local_path):
        return local_path
        
    os.makedirs("data", exist_ok=True)
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=20) as response:
            data = response.read().decode("utf-8")
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(data)
        return local_path
    except Exception as e:
        print("Error downloading Brazil states GeoJSON:", e)
        return None

def add_compass_rose(ax, x=0.12, y=0.20, scale=0.035):
    """
    Draws a 4-point compass rose star pointer on the plot axes,
    with 'N', 'E', 'S', 'W' text labels positioned tight and close to the star pointer.
    """
    # Draw North Star pointer arrow
    ax.annotate('', xy=(x, y + scale), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=6, headlength=7))
    # Draw South arm
    ax.annotate('', xy=(x, y - scale), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='#555555', edgecolor='black', width=1.0, headwidth=4, headlength=5))
    # Draw East arm
    ax.annotate('', xy=(x + scale, y), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='#555555', edgecolor='black', width=1.0, headwidth=4, headlength=5))
    # Draw West arm
    ax.annotate('', xy=(x - scale, y), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='#555555', edgecolor='black', width=1.0, headwidth=4, headlength=5))
    
    # N, E, S, W text labels positioned VERY CLOSE to the star pointer arms
    offset = scale + 0.012
    ax.text(x, y + offset, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
    ax.text(x, y - offset, 'S', transform=ax.transAxes, ha='center', va='top', fontsize=8, fontweight='bold', color='black')
    ax.text(x + offset, y, 'E', transform=ax.transAxes, ha='left', va='center', fontsize=8, fontweight='bold', color='black')
    ax.text(x - offset, y, 'W', transform=ax.transAxes, ha='right', va='center', fontsize=8, fontweight='bold', color='black')

def generate_publication_maps():
    print("=========================================================")
    print("=== GENERATING PUBLICATION MAPS WITH THICK AXES & COMPASS ===")
    print("=========================================================")
    
    geojson_path = download_brazil_geojson()
    if not geojson_path or not os.path.exists(geojson_path):
        print("Skipping spatial map generation (GeoJSON missing).")
        return

    gdf = gpd.read_file(geojson_path)
    state_to_zone = {
        "AC": 1, "AM": 1, "AP": 1, "PA": 1, "RO": 1, "RR": 1, "TO": 1,
        "AL": 2, "BA": 2, "CE": 2, "MA": 2, "PB": 2, "PE": 2, "PI": 2, "RN": 2, "SE": 2,
        "DF": 4, "GO": 4, "MT": 4, "MS": 4,
        "MG": 5, "ES": 5, "RJ": 5, "SP": 5,
        "PR": 6, "RS": 6, "SC": 6
    }
    gdf["climate_zone"] = gdf["sigla"].map(state_to_zone)
    
    # Generate 5-Year and 2-Year Spatial Risk Maps
    for horizon, csv_name, title_text, filename_prefix in [
        ("5years", "final/outputs/csv/zone_dengue_2025_2030.csv", "5-Year Dengue Spatial Risk Projections (2025–2030)", "dengue_forecast_combined_zones_5years"),
        ("2years", "final/outputs_2years/csv/zone_dengue_2025_2026.csv", "2-Year Dengue Spatial Risk Projections (2025–2026)", "dengue_forecast_combined_zones_2years")
    ]:
        if not os.path.exists(csv_name):
            continue
            
        z_df = pd.read_csv(csv_name)
        z_avg = z_df.groupby("climate_zone")["incidence_rate"].mean().to_dict()
        gdf["mean_inc"] = gdf["climate_zone"].map(z_avg)
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
        
        # Plot Brazil map with choropleth shading
        gdf.plot(column="mean_inc", cmap="YlOrRd", linewidth=1.2, edgecolor="#333333", ax=ax, legend=False)
        
        # Add thicker axis border
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color("#000000")
            
        ax.tick_params(colors="#000000", labelsize=10, width=1.8, length=6)
        ax.set_xlabel("Longitude (°W)", fontsize=12, fontweight="bold", fontstyle="italic", labelpad=8)
        ax.set_ylabel("Latitude (°S)", fontsize=12, fontweight="bold", fontstyle="italic", labelpad=8)
        ax.set_title(title_text, fontsize=13, fontweight="bold", pad=12)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=gdf["mean_inc"].min(), vmax=gdf["mean_inc"].max()))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
        cbar.set_label("Mean Forecast Incidence Rate (per 100k)", fontsize=10, fontweight="bold")
        cbar.ax.tick_params(labelsize=9, width=1.5)
        cbar.outline.set_linewidth(1.5)
        
        # Add Compass Rose Star Pointer at bottom left with N, E, S, W close to star
        add_compass_rose(ax, x=0.14, y=0.18, scale=0.035)
        
        plt.tight_layout()
        
        output_dirs = ["figures", "final/outputs/graphs"]
        for d in output_dirs:
            os.makedirs(d, exist_ok=True)
            plt.savefig(os.path.join(d, f"{filename_prefix}.png"), dpi=600, bbox_inches="tight")
            plt.savefig(os.path.join(d, f"{filename_prefix}.pdf"), bbox_inches="tight")
            plt.savefig(os.path.join(d, f"{filename_prefix}.eps"), format="eps", bbox_inches="tight")
            
        plt.close()
        
    print("Publication Maps generated successfully with thick axes & tight compass rose!")

if __name__ == "__main__":
    generate_publication_maps()
