import os
import gc
import ssl
import json
import urllib.request
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

def download_coords():
    local_path = "data/municipios_coords.csv"
    if os.path.exists(local_path):
        print("Using local coordinates file.")
        return pd.read_csv(local_path)
        
    print("Downloading municipality coordinates from GitHub...")
    url = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    os.makedirs("data", exist_ok=True)
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=20) as response:
            data = response.read().decode("utf-8")
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(data)
        print(f"Coordinates saved to {local_path}")
        return pd.read_csv(local_path)
    except Exception as e:
        print("Warning: Could not download coordinates from GitHub. Error:", e)
        # Create a dummy coordinate DataFrame if offline
        print("Creating fallback coordinates (uniform grid) for visualization...")
        # We will generate dummy coords centered around Brazil (-15, -50)
        np.random.seed(42)
        fallback_df = pd.DataFrame({
            "codigo_ibge": [1100015], # Add at least one real IBGE code
            "nome": ["Fallback Municipality"],
            "latitude": [-15.0],
            "longitude": [-50.0]
        })
        return fallback_df

def generate_maps():
    print("Initializing map generation...")
    
    # Load coordinates
    coords_df = download_coords()
    coords_df = coords_df.rename(columns={"codigo_ibge": "geocode", "nome": "mun_name"})
    
    # Load forecasts
    m_dengue_path = "final/outputs/csv/municipality_dengue_2025_2029.csv"
    m_climate_path = "final/outputs/csv/municipality_climate_2025_2029.csv"
    z_dengue_path = "final/outputs/csv/zone_dengue_2025_2029.csv"
    
    if not (os.path.exists(m_dengue_path) and os.path.exists(m_climate_path) and os.path.exists(z_dengue_path)):
        raise FileNotFoundError("Forecast CSV files missing in final/outputs/csv/. Run prediction scripts first.")
        
    print("Loading forecasts for mapping...")
    m_dengue = pd.read_csv(m_dengue_path)
    m_climate = pd.read_csv(m_climate_path)
    z_dengue = pd.read_csv(z_dengue_path)
    
    # Convert dates to datetime
    m_dengue["date"] = pd.to_datetime(m_dengue["date"])
    m_climate["date"] = pd.to_datetime(m_climate["date"])
    z_dengue["date"] = pd.to_datetime(z_dengue["date"])
    
    # Merge coords
    m_dengue = m_dengue.merge(coords_df[["geocode", "mun_name", "latitude", "longitude"]], on="geocode", how="inner")
    m_climate = m_climate.merge(coords_df[["geocode", "mun_name", "latitude", "longitude"]], on="geocode", how="inner")
    
    print(f"Data ready. Dengue map rows: {len(m_dengue)}, Climate map rows: {len(m_climate)}")
    
    # ----------------------------------------------------
    # 1. MUNICIPALITY DENGUE ANIMATION (Monthly Aggregated)
    # ----------------------------------------------------
    print("Generating Municipality Dengue Animation...")
    # Aggregate weekly to monthly to keep HTML file lightweight and animations fluid
    m_dengue["year_month"] = m_dengue["date"].dt.strftime("%Y-%m")
    m_dengue_monthly = m_dengue.groupby(["year_month", "geocode", "mun_name", "latitude", "longitude", "uf", "climate_zone", "population"], as_index=False).agg({
        "cases": "sum",
        "incidence_rate": "mean"
    }).sort_values("year_month")
    
    fig_dengue = px.scatter_geo(
        m_dengue_monthly,
        lat="latitude",
        lon="longitude",
        color="incidence_rate",
        size="cases",
        hover_name="mun_name",
        hover_data={
            "year_month": True,
            "uf": True,
            "climate_zone": True,
            "population": ":,.0f",
            "cases": ":,.0f",
            "incidence_rate": ":.2f"
        },
        animation_frame="year_month",
        projection="mercator",
        color_continuous_scale="Reds",
        range_color=[0, m_dengue_monthly["incidence_rate"].quantile(0.95)],
        size_max=25,
        title="Brazil Municipality Dengue Cases & Incidence Forecast (2025-2029)"
    )
    
    # Apply elegant dark style layout
    fig_dengue.update_geos(
        showcountries=True, countrycolor="dimgray",
        showland=True, landcolor="#1e1e1e",
        showocean=True, oceancolor="#121212",
        showlakes=False,
        fitbounds="locations"
    )
    fig_dengue.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Incidence Rate<br>(per 100k)")
    )
    
    html_dengue_path = "final/outputs/maps/municipality_dengue_animation.html"
    fig_dengue.write_html(html_dengue_path)
    print(f"Saved Dengue map to {html_dengue_path}")
    
    # Save PNG Snapshot (first frame)
    try:
        fig_dengue.write_image("final/outputs/maps/municipality_dengue_snapshot.png", scale=2)
        print("Saved Dengue map PNG snapshot.")
    except Exception as e:
        print("Note: Skipping static PNG export. Install kaleido to export PNGs.")
        
    del m_dengue_monthly, fig_dengue
    gc.collect()
    
    # ----------------------------------------------------
    # 2. MUNICIPALITY CLIMATE ANIMATION (Monthly Aggregated)
    # ----------------------------------------------------
    print("Generating Municipality Climate Animation...")
    m_climate["year_month"] = m_climate["date"].dt.strftime("%Y-%m")
    m_climate_monthly = m_climate.groupby(["year_month", "geocode", "mun_name", "latitude", "longitude", "uf", "climate_zone"], as_index=False).agg({
        "temp_med": "mean",
        "precip_tot": "sum"
    }).sort_values("year_month")
    
    # Clip variables to be non-negative to avoid negative sizes in Plotly scatter_geo
    m_climate_monthly["precip_tot"] = m_climate_monthly["precip_tot"].clip(lower=0.0)
    m_climate_monthly["temp_med"] = m_climate_monthly["temp_med"].clip(lower=0.0)
    
    fig_climate = px.scatter_geo(
        m_climate_monthly,
        lat="latitude",
        lon="longitude",
        color="temp_med",
        size="precip_tot",
        hover_name="mun_name",
        hover_data={
            "year_month": True,
            "uf": True,
            "climate_zone": True,
            "temp_med": ":.1f",
            "precip_tot": ":.1f"
        },
        animation_frame="year_month",
        projection="mercator",
        color_continuous_scale="Thermal",
        range_color=[m_climate_monthly["temp_med"].min(), m_climate_monthly["temp_med"].max()],
        size_max=15,
        title="Brazil Municipality Temperature & Precipitation Forecast (2025-2029)"
    )
    fig_climate.update_geos(
        showcountries=True, countrycolor="dimgray",
        showland=True, landcolor="#1e1e1e",
        showocean=True, oceancolor="#121212",
        showlakes=False,
        fitbounds="locations"
    )
    fig_climate.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Temp (°C)")
    )
    
    html_climate_path = "final/outputs/maps/municipality_climate_animation.html"
    fig_climate.write_html(html_climate_path)
    print(f"Saved Climate map to {html_climate_path}")
    
    try:
        fig_climate.write_image("final/outputs/maps/municipality_climate_snapshot.png", scale=2)
        print("Saved Climate map PNG snapshot.")
    except Exception:
        pass
        
    del m_climate_monthly, fig_climate
    gc.collect()
    
    # ----------------------------------------------------
    # 3. CLIMATE ZONE DENGUE ANIMATION (Weekly Resolution)
    # ----------------------------------------------------
    print("Generating Climate Zone Dengue Animation...")
    # Map municipalities to their climate zones, then join weekly zone dengue values
    m_zones = m_dengue[["geocode", "mun_name", "latitude", "longitude", "climate_zone"]].drop_duplicates(subset=["geocode"])
    
    # Merge zones with zone dengue predictions weekly
    # z_dengue has columns: date, year, week, climate_zone, population, cases, incidence_rate
    # We want a dataframe of all municipalities, weekly, colored by their zone's incidence rate
    z_dengue["date_str"] = z_dengue["date"].dt.strftime("%Y-%m-%d")
    z_dengue_sorted = z_dengue.sort_values("date")
    
    # To avoid gigabytes of dataframe replication, let's keep only a subset of columns and sample weekly
    # Since there are 5561 municipalities, 260 weeks is 1.45M rows.
    # We will merge them and sort.
    zone_map_df = z_dengue_sorted[["date_str", "climate_zone", "cases", "incidence_rate"]].merge(m_zones, on="climate_zone", how="inner")
    
    fig_zones = px.scatter_geo(
        zone_map_df,
        lat="latitude",
        lon="longitude",
        color="incidence_rate",
        hover_name="mun_name",
        hover_data={
            "date_str": True,
            "climate_zone": True,
            "incidence_rate": ":.2f"
        },
        animation_frame="date_str",
        projection="mercator",
        color_continuous_scale="Portland",
        range_color=[0, zone_map_df["incidence_rate"].quantile(0.95)],
        title="Brazil Climate Zone Weekly Dengue Incidence Forecast (2025-2029)"
    )
    fig_zones.update_geos(
        showcountries=True, countrycolor="dimgray",
        showland=True, landcolor="#1e1e1e",
        showocean=True, oceancolor="#121212",
        showlakes=False,
        fitbounds="locations"
    )
    fig_zones.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Zone Incidence<br>(per 100k)")
    )
    
    html_zone_path = "final/outputs/maps/climate_zone_dengue_animation.html"
    fig_zones.write_html(html_zone_path)
    print(f"Saved Climate Zone map to {html_zone_path}")
    
    try:
        fig_zones.write_image("final/outputs/maps/climate_zone_dengue_snapshot.png", scale=2)
        print("Saved Climate Zone map PNG snapshot.")
    except Exception:
        pass
        
    del zone_map_df, fig_zones
    gc.collect()
    
    # ----------------------------------------------------
    # 4. STATE DENGUE CHOROPLETH ANIMATION (Weekly Resolution)
    # ----------------------------------------------------
    print("Generating State-wise Choropleth map...")
    geojson_path = "data/brazil_states.geojson"
    if not os.path.exists(geojson_path):
        print("Downloading Brazil states GeoJSON...")
        url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=20) as response:
                data = response.read().decode("utf-8")
                with open(geojson_path, "w", encoding="utf-8") as f:
                    f.write(data)
            print(f"GeoJSON saved to {geojson_path}")
        except Exception as e:
            print("Error downloading GeoJSON:", e)
            
    if os.path.exists(geojson_path):
        with open(geojson_path, "r", encoding="utf-8") as f:
            brazil_geojson = json.load(f)
            
        for feature in brazil_geojson["features"]:
            feature["id"] = feature["properties"]["sigla"]
            
        m_dengue["year_month"] = m_dengue["date"].dt.strftime("%Y-%m")
        state_df = m_dengue.groupby(["year_month", "uf"], as_index=False).agg({
            "cases": "sum",
            "population": "first"
        }).sort_values("year_month")
        state_df["incidence_rate"] = (state_df["cases"] / state_df["population"] * 100000.0).astype(np.float32)
        
        fig_states = px.choropleth(
            state_df,
            geojson=brazil_geojson,
            locations="uf",
            color="incidence_rate",
            hover_name="uf",
            hover_data={
                "year_month": True,
                "population": ":,.0f",
                "cases": ":,.0f",
                "incidence_rate": ":.2f"
            },
            animation_frame="year_month",
            color_continuous_scale="Reds",
            range_color=[0, state_df["incidence_rate"].quantile(0.95)],
            title="Brazil State-wise Dengue Incidence Forecast (2025-2029)"
        )
        fig_states.update_geos(
            fitbounds="locations",
            visible=False
        )
        fig_states.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(title="Incidence Rate<br>(per 100k)")
        )
        
        html_states_path = "final/outputs/maps/state_dengue_forecast_animation.html"
        fig_states.write_html(html_states_path)
        print(f"Saved State-wise Choropleth map to {html_states_path}")
        
        try:
            fig_states.write_image("final/outputs/maps/state_dengue_forecast_snapshot.png", scale=2)
            print("Saved State-wise map PNG snapshot.")
        except Exception:
            pass
            
        # ----------------------------------------------------
        # 5. CLIMATE ZONE CHOROPLETH ANIMATION (Monthly Resolution)
        # ----------------------------------------------------
        print("Generating Climate Zone Choropleth map...")
        state_to_zone = {
            "AC": 1, "AM": 1, "AP": 1, "PA": 1, "RO": 1, "RR": 1,
            "MA": 2, "TO": 2,
            "AL": 3, "CE": 3, "PB": 3, "PE": 3, "PI": 3, "RN": 3, "SE": 3,
            "BA": 4, "DF": 4, "GO": 4, "MS": 4, "MT": 4,
            "ES": 5, "MG": 5, "RJ": 5, "SP": 5,
            "PR": 6, "RS": 6, "SC": 6
        }
        
        z_dengue["year_month"] = z_dengue["date"].dt.strftime("%Y-%m")
        z_monthly = z_dengue.groupby(["year_month", "climate_zone"], as_index=False).agg({
            "cases": "sum",
            "population": "first"
        })
        z_monthly["incidence_rate"] = (z_monthly["cases"] / z_monthly["population"] * 100000.0).astype(np.float32)
        
        records = []
        for _, row in z_monthly.iterrows():
            ym = row["year_month"]
            zone = int(row["climate_zone"])
            inc = row["incidence_rate"]
            cases = row["cases"]
            
            states_in_zone = [uf for uf, z in state_to_zone.items() if z == zone]
            for uf in states_in_zone:
                records.append({
                    "year_month": ym,
                    "uf": uf,
                    "climate_zone": f"Zone {zone}",
                    "zone_incidence": inc,
                    "zone_cases": cases
                })
                
        zone_state_df = pd.DataFrame(records).sort_values("year_month")
        
        fig_zone_choropleth = px.choropleth(
            zone_state_df,
            geojson=brazil_geojson,
            locations="uf",
            color="zone_incidence",
            hover_name="uf",
            hover_data={
                "year_month": True,
                "climate_zone": True,
                "zone_cases": ":,.0f",
                "zone_incidence": ":.2f"
            },
            animation_frame="year_month",
            color_continuous_scale="Portland",
            range_color=[0, zone_state_df["zone_incidence"].quantile(0.95)],
            title="Brazil Climate Zone Dengue Incidence Forecast (2025-2029)"
        )
        fig_zone_choropleth.update_geos(
            fitbounds="locations",
            visible=False
        )
        fig_zone_choropleth.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(title="Zone Incidence<br>(per 100k)")
        )
        
        html_zone_choro_path = "final/outputs/maps/climate_zone_dengue_choropleth.html"
        fig_zone_choropleth.write_html(html_zone_choro_path)
        print(f"Saved Climate Zone Choropleth map to {html_zone_choro_path}")
        
        try:
            fig_zone_choropleth.write_image("final/outputs/maps/climate_zone_dengue_choropleth_snapshot.png", scale=2)
            print("Saved Climate Zone Choropleth map PNG snapshot.")
        except Exception:
            pass
            
        del state_df, fig_states, zone_state_df, fig_zone_choropleth
        gc.collect()
        
    print("Map generation finished. Interactive maps saved to final/outputs/maps/")

if __name__ == "__main__":
    generate_maps()
