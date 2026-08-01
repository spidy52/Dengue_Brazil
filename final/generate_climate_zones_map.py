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
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch, Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.dpi": 600,
    "savefig.dpi": 600
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
        print("Error downloading GeoJSON:", e)
        return None

def generate_climate_zones_map():
    print("=========================================================")
    print("=== GENERATING BRAZIL 6 CLIMATE ZONES DIAGRAM MAP ===")
    print("=========================================================")
    
    geojson_path = download_brazil_geojson()
    if not geojson_path or not os.path.exists(geojson_path):
        print("GeoJSON missing.")
        return

    gdf = gpd.read_file(geojson_path)
    # Simplify Brazil state polygon vertices to optimize EPS vector file size while maintaining crisp borders
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)
    
    # Exact 6 Climate Zones color scheme matching Image 2
    zone_colors = {
        1: "#3e9668",  # Zone 1 - Equatorial Amazon (Green)
        2: "#f68b33",  # Zone 2 - Cerrado North (Orange)
        3: "#f06d73",  # Zone 3 - Semi-Arid Northeast (Pink/Red)
        4: "#a572b7",  # Zone 4 - Central-West (Purple)
        5: "#4aa0e6",  # Zone 5 - Southeast Core (Blue)
        6: "#f7c828"   # Zone 6 - Southern Temperate (Yellow)
    }
    
    state_to_zone = {
        "AC": 1, "AM": 1, "AP": 1, "PA": 1, "RO": 1, "RR": 1,
        "MA": 2, "TO": 2,
        "AL": 3, "CE": 3, "PB": 3, "PE": 3, "PI": 3, "RN": 3, "SE": 3,
        "BA": 4, "DF": 4, "GO": 4, "MS": 4, "MT": 4,
        "ES": 5, "MG": 5, "RJ": 5, "SP": 5,
        "PR": 6, "RS": 6, "SC": 6
    }
    gdf["climate_zone"] = gdf["sigla"].map(state_to_zone)
    gdf["color"] = gdf["climate_zone"].map(zone_colors)
    
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    
    # Grid spec: Left for World Map, Right for Brazil 6 Climate Zones Map
    ax_world = fig.add_axes([0.02, 0.25, 0.35, 0.50])
    ax_brazil = fig.add_axes([0.42, 0.05, 0.55, 0.90])
    
    # ------------------ LEFT PANEL: WORLD MAP INSET ------------------
    world_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(world_url, context=ctx, timeout=20) as r:
            world_gdf = gpd.read_file(r)
            
        # Simplify world map polygons for lightweight EPS export
        world_gdf["geometry"] = world_gdf["geometry"].simplify(tolerance=0.08, preserve_topology=True)
        world_gdf.plot(ax=ax_world, color="#e5e7eb", edgecolor="#ffffff", lw=0.5)
    except Exception as e:
        print("World map fetch notice:", e)
        
    # Highlight Brazil as a single solid green (#3e9668) country shape on World Map inset (no internal state lines)
    gdf.dissolve().plot(ax=ax_world, color="#3e9668", edgecolor="none", zorder=5)
        
    ax_world.set_xlim(-170, 180)
    ax_world.set_ylim(-60, 85)
    ax_world.axis("off")
    
    # World Map title badge
    ax_world.text(-60, 55, "  WORLD MAP  ", bbox=dict(boxstyle="round,pad=0.4", facecolor="#0b1d3a", edgecolor="none"),
                  color="white", fontsize=11, fontweight="bold", ha="center", va="center")
                  
    # Dashed bounding box around South America
    rect = Rectangle((-85, -38), 50, 48, fill=False, edgecolor="#0b1d3a", linestyle="--", linewidth=1.8)
    ax_world.add_patch(rect)
    
    # Callout connecting lines from World Map box to Brazil Map container
    con1 = patches.ConnectionPatch(xyA=(-35, 10), xyB=(-0.02, 0.98), coordsA="data", coordsB="axes fraction",
                                  axesA=ax_world, axesB=ax_brazil, color="#0b1d3a", lw=1.8)
    con2 = patches.ConnectionPatch(xyA=(-35, -38), xyB=(-0.02, 0.02), coordsA="data", coordsB="axes fraction",
                                  axesA=ax_world, axesB=ax_brazil, color="#0b1d3a", lw=1.8)
    fig.add_artist(con1)
    fig.add_artist(con2)
    
    # ------------------ RIGHT PANEL: BRAZIL 6 CLIMATE ZONES ------------------
    for _, row in gdf.iterrows():
        gpd.GeoSeries([row.geometry]).plot(ax=ax_brazil, color=row["color"], edgecolor="#ffffff", linewidth=1.5)
        
        # State label
        centroid = row.geometry.centroid
        ax_brazil.text(centroid.x, centroid.y, row["sigla"], fontsize=8.5, fontweight="bold", color="#111111", ha="center", va="center")
        
    ax_brazil.set_xlim(-74, -33)
    ax_brazil.set_ylim(-34, 6)
    
    # Container box around Brazil Map with dark navy border
    for spine in ax_brazil.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("#0b1d3a")
        
    ax_brazil.set_xticks([])
    ax_brazil.set_yticks([])
    
    # Top title badge inside Brazil map container
    ax_brazil.text(0.50, 0.96, "  BRAZIL – 6 CLIMATE ZONES  ", transform=ax_brazil.transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#0b1d3a", edgecolor="none"),
                   color="white", fontsize=13, fontweight="bold", ha="center", va="top", zorder=20)
                   
    # Legend Box at Bottom-Left inside Brazil Map
    legend_text = (
        "■ Zone 1 – Equatorial Amazon\n   (AC, AM, AP, PA, RO, RR)\n\n"
        "■ Zone 2 – Cerrado North\n   (MA, TO)\n\n"
        "■ Zone 3 – Semi-Arid Northeast\n   (AL, CE, PB, PE, PI, RN, SE)\n\n"
        "■ Zone 4 – Central-West\n   (BA, DF, GO, MS, MT)\n\n"
        "■ Zone 5 – Southeast Core\n   (ES, MG, RJ, SP)\n\n"
        "■ Zone 6 – Southern Temperate\n   (PR, RS, SC)"
    )
    
    # Draw legend box with zone entries
    lx, ly = 0.05, 0.04
    ax_brazil.text(lx, ly, legend_text, transform=ax_brazil.transAxes, fontsize=8.0, fontweight="bold",
                   color="#222222", va="bottom", ha="left",
                   bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.95), zorder=20)

    # ------------------ COMPASS ROSE STAR & SCALE BAR (BOTTOM RIGHT) ------------------
    cx, cy = 0.88, 0.16
    size = 0.040
    inner = size * 0.22
    
    # 4-Point Shaded Star (Navy & White)
    poly_N_E = MplPolygon([[cx, cy], [cx, cy + size], [cx + inner, cy + inner]], facecolor='#0b1d3a', edgecolor='#0b1d3a', lw=0.8, transform=ax_brazil.transAxes, zorder=21)
    poly_N_W = MplPolygon([[cx, cy], [cx, cy + size], [cx - inner, cy + inner]], facecolor='#ffffff', edgecolor='#0b1d3a', lw=0.8, transform=ax_brazil.transAxes, zorder=21)
    
    poly_S_E = MplPolygon([[cx, cy], [cx, cy - size], [cx + inner, cy - inner]], facecolor='#ffffff', edgecolor='#0b1d3a', lw=0.8, transform=ax_brazil.transAxes, zorder=21)
    poly_S_W = MplPolygon([[cx, cy], [cx, cy - size], [cx - inner, cy - inner]], facecolor='#0b1d3a', edgecolor='#0b1d3a', lw=0.8, transform=ax_brazil.transAxes, zorder=21)

    poly_E_N = MplPolygon([[cx, cy], [cx + size, cy], [cx + inner, cy + inner]], facecolor='#ffffff', edgecolor='#0b1d3a', lw=0.8, transform=ax_brazil.transAxes, zorder=21)
    poly_E_S = MplPolygon([[cx, cy], [cx + size, cy], [cx + inner, cy - inner]], facecolor='#0b1d3a', edgecolor='#0b1d3a', lw=0.8, transform=ax_brazil.transAxes, zorder=21)

    poly_W_N = MplPolygon([[cx, cy], [cx - size, cy], [cx - inner, cy + inner]], facecolor='#0b1d3a', edgecolor='#0b1d3a', lw=0.8, transform=ax_brazil.transAxes, zorder=21)
    poly_W_S = MplPolygon([[cx, cy], [cx - size, cy], [cx - inner, cy - inner]], facecolor='#ffffff', edgecolor='#0b1d3a', lw=0.8, transform=ax_brazil.transAxes, zorder=21)

    for p in [poly_N_E, poly_N_W, poly_S_E, poly_S_W, poly_E_N, poly_E_S, poly_W_N, poly_W_S]:
        ax_brazil.add_patch(p)

    # N, E, S, W text labels positioned VERY CLOSE to star tips
    off = size + 0.0012
    ax_brazil.text(cx, cy + off, 'N', transform=ax_brazil.transAxes, ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0b1d3a', zorder=22)
    ax_brazil.text(cx, cy - off, 'S', transform=ax_brazil.transAxes, ha='center', va='top', fontsize=9.0, fontweight='bold', color='#0b1d3a', zorder=22)
    ax_brazil.text(cx + off, cy, 'E', transform=ax_brazil.transAxes, ha='left', va='center', fontsize=9.0, fontweight='bold', color='#0b1d3a', zorder=22)
    ax_brazil.text(cx - off, cy, 'W', transform=ax_brazil.transAxes, ha='right', va='center', fontsize=9.0, fontweight='bold', color='#0b1d3a', zorder=22)

    # Scale Bar (0, 500, 1000 km) below compass
    sb_y = cy - off - 0.050
    sb_w = 0.050
    ax_brazil.plot([cx - sb_w, cx + sb_w], [sb_y, sb_y], color='#0b1d3a', lw=2.0, transform=ax_brazil.transAxes, zorder=22)
    ax_brazil.plot([cx - sb_w, cx - sb_w], [sb_y, sb_y + 0.008], color='#0b1d3a', lw=1.5, transform=ax_brazil.transAxes, zorder=22)
    ax_brazil.plot([cx, cx], [sb_y, sb_y + 0.008], color='#0b1d3a', lw=1.5, transform=ax_brazil.transAxes, zorder=22)
    ax_brazil.plot([cx + sb_w, cx + sb_w], [sb_y, sb_y + 0.008], color='#0b1d3a', lw=1.5, transform=ax_brazil.transAxes, zorder=22)
    
    ax_brazil.text(cx - sb_w, sb_y - 0.012, '0', transform=ax_brazil.transAxes, ha='center', va='top', fontsize=8, fontweight='bold', color='#0b1d3a', zorder=22)
    ax_brazil.text(cx, sb_y - 0.012, '500', transform=ax_brazil.transAxes, ha='center', va='top', fontsize=8, fontweight='bold', color='#0b1d3a', zorder=22)
    ax_brazil.text(cx + sb_w, sb_y - 0.012, '1000 km', transform=ax_brazil.transAxes, ha='center', va='top', fontsize=8, fontweight='bold', color='#0b1d3a', zorder=22)

    plt.tight_layout()
    
    output_dirs = ["figures", "final/outputs/graphs"]
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, "brazil_6_climate_zones_map.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(d, "brazil_6_climate_zones_map.pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(d, "brazil_6_climate_zones_map.eps"), format="eps", bbox_inches="tight")
        
    plt.close()
    print("Brazil 6 Climate Zones Diagram Map generated successfully!")

if __name__ == "__main__":
    generate_climate_zones_map()
