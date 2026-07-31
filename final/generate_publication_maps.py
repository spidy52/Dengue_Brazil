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

def add_compass_rose(ax, x=0.22, y=0.24, size=0.045):
    """
    Draws an authentic 4-pointed shaded star compass rose with N, E, S, W
    text labels positioned tight and close to the star points, plus a 500-1000 km scale bar.
    """
    inner = size * 0.22

    # Polygons for 4-point shaded star
    poly_N_E = MplPolygon([[x, y], [x, y + size], [x + inner, y + inner]], facecolor='#0b1d3a', edgecolor='#000000', lw=0.8, transform=ax.transAxes, zorder=10)
    poly_N_W = MplPolygon([[x, y], [x, y + size], [x - inner, y + inner]], facecolor='#ffffff', edgecolor='#000000', lw=0.8, transform=ax.transAxes, zorder=10)
    
    poly_S_E = MplPolygon([[x, y], [x, y - size], [x + inner, y - inner]], facecolor='#ffffff', edgecolor='#000000', lw=0.8, transform=ax.transAxes, zorder=10)
    poly_S_W = MplPolygon([[x, y], [x, y - size], [x - inner, y - inner]], facecolor='#0b1d3a', edgecolor='#000000', lw=0.8, transform=ax.transAxes, zorder=10)

    poly_E_N = MplPolygon([[x, y], [x + size, y], [x + inner, y + inner]], facecolor='#ffffff', edgecolor='#000000', lw=0.8, transform=ax.transAxes, zorder=10)
    poly_E_S = MplPolygon([[x, y], [x + size, y], [x + inner, y - inner]], facecolor='#0b1d3a', edgecolor='#000000', lw=0.8, transform=ax.transAxes, zorder=10)

    poly_W_N = MplPolygon([[x, y], [x - size, y], [x - inner, y + inner]], facecolor='#0b1d3a', edgecolor='#000000', lw=0.8, transform=ax.transAxes, zorder=10)
    poly_W_S = MplPolygon([[x, y], [x - size, y], [x - inner, y - inner]], facecolor='#ffffff', edgecolor='#000000', lw=0.8, transform=ax.transAxes, zorder=10)

    for p in [poly_N_E, poly_N_W, poly_S_E, poly_S_W, poly_E_N, poly_E_S, poly_W_N, poly_W_S]:
        ax.add_patch(p)

    # N, E, S, W text labels positioned EXTREMELY CLOSE (hugging) to the star tips
    off = size + 0.0015
    ax.text(x, y + off, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='black', zorder=11)
    ax.text(x, y - off, 'S', transform=ax.transAxes, ha='center', va='top', fontsize=8.0, fontweight='bold', color='black', zorder=11)
    ax.text(x + off, y, 'E', transform=ax.transAxes, ha='left', va='center', fontsize=8.0, fontweight='bold', color='black', zorder=11)
    ax.text(x - off, y, 'W', transform=ax.transAxes, ha='right', va='center', fontsize=8.0, fontweight='bold', color='black', zorder=11)

    # Scale bar below 'S'
    sb_y = y - off - 0.035
    ax.plot([x - 0.03, x + 0.03], [sb_y, sb_y], color='black', lw=1.5, transform=ax.transAxes, zorder=11)
    ax.plot([x - 0.03, x - 0.03], [sb_y, sb_y + 0.006], color='black', lw=1.2, transform=ax.transAxes, zorder=11)
    ax.plot([x, x], [sb_y, sb_y + 0.006], color='black', lw=1.2, transform=ax.transAxes, zorder=11)
    ax.plot([x + 0.03, x + 0.03], [sb_y, sb_y + 0.006], color='black', lw=1.2, transform=ax.transAxes, zorder=11)
    
    ax.text(x - 0.03, sb_y - 0.010, '0', transform=ax.transAxes, ha='center', va='top', fontsize=7.5, fontweight='bold', color='black', zorder=11)
    ax.text(x, sb_y - 0.010, '500', transform=ax.transAxes, ha='center', va='top', fontsize=7.5, fontweight='bold', color='black', zorder=11)
    ax.text(x + 0.03, sb_y - 0.010, '1000 km', transform=ax.transAxes, ha='center', va='top', fontsize=7.5, fontweight='bold', color='black', zorder=11)

def generate_publication_maps():
    print("=========================================================")
    print("=== GENERATING PUBLICATION MAPS WITH THICK AXES & TIGHT COMPASS ===")
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
        
        # Add state acronym annotations (e.g. RJ, SP, MG, BA, AM)
        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            ax.text(centroid.x, centroid.y, row["sigla"], fontsize=7.5, fontweight="bold", color="#111111", ha="center", va="center", zorder=8)
        
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
        
        # Add Compass Rose Star Pointer with N, E, S, W tight to star and 500 1000 km scale bar
        add_compass_rose(ax, x=0.12, y=0.20, size=0.032)
        
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
