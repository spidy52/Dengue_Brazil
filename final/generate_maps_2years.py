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
        return pd.read_csv(local_path)
        
    print("Downloading coordinates...")
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
        return pd.read_csv(local_path)
    except Exception:
        fallback_df = pd.DataFrame({
            "codigo_ibge": [1100015],
            "nome": ["Fallback"],
            "latitude": [-15.0],
            "longitude": [-50.0]
        })
        return fallback_df

def generate_maps_2years():
    print("Initializing 2-Year map generation...")
    
    coords_df = download_coords()
    coords_df = coords_df.rename(columns={"codigo_ibge": "geocode", "nome": "mun_name"})
    
    m_dengue_path = "final/outputs_2years/csv/municipality_dengue_2025_2026.csv"
    m_climate_path = "final/outputs/csv/municipality_climate_2025_2029.csv"
    z_dengue_path = "final/outputs_2years/csv/zone_dengue_2025_2026.csv"
    
    if not (os.path.exists(m_dengue_path) and os.path.exists(m_climate_path) and os.path.exists(z_dengue_path)):
        raise FileNotFoundError("Forecast CSV files missing.")
        
    m_dengue = pd.read_csv(m_dengue_path)
    m_climate = pd.read_csv(m_climate_path)
    m_climate["date"] = pd.to_datetime(m_climate["date"])
    m_climate = m_climate[m_climate["date"] <= "2026-12-31"] # Filter climate to 2 years
    
    z_dengue = pd.read_csv(z_dengue_path)
    
    m_dengue["date"] = pd.to_datetime(m_dengue["date"])
    z_dengue["date"] = pd.to_datetime(z_dengue["date"])
    
    m_dengue = m_dengue.merge(coords_df[["geocode", "mun_name", "latitude", "longitude"]], on="geocode", how="inner")
    m_climate = m_climate.merge(coords_df[["geocode", "mun_name", "latitude", "longitude"]], on="geocode", how="inner")
    
    os.makedirs("final/outputs_2years/maps", exist_ok=True)
    
    # 1. Municipality Dengue Animation
    print("Generating Municipality Dengue Animation...")
    m_dengue["year_month"] = m_dengue["date"].dt.strftime("%Y-%m")
    m_dengue_monthly = m_dengue.groupby(["year_month", "geocode", "mun_name", "latitude", "longitude", "uf", "climate_zone", "population"], as_index=False).agg({
        "cases": "sum",
        "incidence_rate": "mean"
    }).sort_values("year_month")
    
    m_dengue_monthly["cases"] = m_dengue_monthly["cases"].clip(lower=0.0)
    m_dengue_monthly["incidence_rate"] = m_dengue_monthly["incidence_rate"].clip(lower=0.0)
    
    fig_dengue = px.scatter_geo(
        m_dengue_monthly,
        lat="latitude",
        lon="longitude",
        color="incidence_rate",
        size="cases",
        hover_name="mun_name",
        hover_data={"year_month": True, "uf": True, "climate_zone": True, "population": ":,.0f", "cases": ":,.0f", "incidence_rate": ":.2f"},
        animation_frame="year_month",
        projection="natural earth",
        title="Brazil Municipality Dengue Forecast (2025-2026)",
        color_continuous_scale="Jet",
        range_color=[0, m_dengue_monthly["incidence_rate"].quantile(0.95)]
    )
    fig_dengue.update_geos(scope="south america", showcountries=True, countrycolor="gray", showland=True, landcolor="#111111", showocean=True, oceancolor="#000000")
    fig_dengue.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar=dict(title="Incidence Rate<br>(per 100k)"))
    
    html_dengue_path = "final/outputs_2years/maps/municipality_dengue_animation.html"
    fig_dengue.write_html(html_dengue_path)
    print(f"Saved Dengue map to {html_dengue_path}")
    
    # 2. Municipality Climate Animation
    print("Generating Municipality Climate Animation...")
    m_climate["year_month"] = m_climate["date"].dt.strftime("%Y-%m")
    m_climate_monthly = m_climate.groupby(["year_month", "geocode", "mun_name", "latitude", "longitude", "uf", "climate_zone"], as_index=False).agg({
        "temp_med": "mean",
        "precip_tot": "sum"
    }).sort_values("year_month")
    
    # Clip climate variables to be non-negative
    m_climate_monthly["precip_tot"] = m_climate_monthly["precip_tot"].clip(lower=0.0)
    m_climate_monthly["temp_med"] = m_climate_monthly["temp_med"].clip(lower=0.0)
    
    fig_climate = px.scatter_geo(
        m_climate_monthly,
        lat="latitude",
        lon="longitude",
        color="temp_med",
        size="precip_tot",
        hover_name="mun_name",
        hover_data={"year_month": True, "uf": True, "climate_zone": True, "temp_med": ":.2f", "precip_tot": ":.2f"},
        animation_frame="year_month",
        projection="natural earth",
        title="Brazil Municipality Temperature & Rainfall Forecast (2025-2026)",
        color_continuous_scale="Reds",
        range_color=[15, 35]
    )
    fig_climate.update_geos(scope="south america", showcountries=True, countrycolor="gray", showland=True, landcolor="#111111", showocean=True, oceancolor="#000000")
    fig_climate.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar=dict(title="Temp (°C)"))
    
    html_climate_path = "final/outputs_2years/maps/municipality_climate_animation.html"
    fig_climate.write_html(html_climate_path)
    print(f"Saved Climate map to {html_climate_path}")
    
    # 3. Climate Zone Dengue Animation
    print("Generating Climate Zone Dengue Animation...")
    z_dengue["year_month"] = z_dengue["date"].dt.strftime("%Y-%m")
    z_dengue_monthly = z_dengue.groupby(["year_month", "climate_zone", "population"], as_index=False).agg({
        "cases": "sum"
    }).sort_values("year_month")
    z_dengue_monthly["cases"] = z_dengue_monthly["cases"].clip(lower=0.0)
    z_dengue_monthly["incidence_rate"] = (z_dengue_monthly["cases"] / z_dengue_monthly["population"] * 100000.0).astype(np.float32)
    
    zone_coords = m_dengue.groupby("climate_zone")[["latitude", "longitude"]].mean().reset_index()
    z_dengue_monthly = z_dengue_monthly.merge(zone_coords, on="climate_zone", how="inner")
    
    fig_zone = px.scatter_geo(
        z_dengue_monthly,
        lat="latitude",
        lon="longitude",
        color="incidence_rate",
        size="cases",
        hover_name="climate_zone",
        hover_data={"year_month": True, "cases": ":,.0f", "incidence_rate": ":.2f"},
        animation_frame="year_month",
        projection="natural earth",
        title="Brazil Climate Zone Dengue Forecast (2025-2026)",
        color_continuous_scale="Bluered",
        range_color=[0, z_dengue_monthly["incidence_rate"].max()]
    )
    fig_zone.update_geos(scope="south america", showcountries=True, countrycolor="gray", showland=True, landcolor="#111111", showocean=True, oceancolor="#000000")
    fig_zone.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar=dict(title="Zone Incidence<br>(per 100k)"))
    
    html_zone_path = "final/outputs_2years/maps/climate_zone_dengue_animation.html"
    fig_zone.write_html(html_zone_path)
    print(f"Saved Climate Zone map to {html_zone_path}")
    
    # 4. State-wise Choropleth and Climate Zone Choropleth
    print("Generating Choropleth Maps...")
    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(geojson_url, context=ctx, timeout=20) as r:
            brazil_geojson = json.loads(r.read().decode("utf-8"))
            
        state_df = m_dengue_monthly.groupby(["year_month", "uf"], as_index=False).agg({
            "cases": "sum",
            "population": "sum"
        })
        state_df["incidence_rate"] = (state_df["cases"] / state_df["population"] * 100000.0).astype(np.float32)
        
        fig_states = px.choropleth(
            state_df,
            geojson=brazil_geojson,
            locations="uf",
            color="incidence_rate",
            featureidkey="properties.sigla",
            hover_name="uf",
            hover_data={"year_month": True, "cases": ":,.0f", "incidence_rate": ":.2f"},
            animation_frame="year_month",
            color_continuous_scale="Reds",
            range_color=[0, state_df["incidence_rate"].quantile(0.95)],
            title="Brazil State-wise Dengue Incidence Forecast (2025-2026)"
        )
        fig_states.update_geos(fitbounds="locations", visible=False)
        fig_states.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar=dict(title="State Incidence<br>(per 100k)"))
        
        html_states_path = "final/outputs_2years/maps/state_dengue_forecast_animation.html"
        fig_states.write_html(html_states_path)
        print(f"Saved State-wise Choropleth map to {html_states_path}")
        
        # Climate Zone Choropleth
        state_to_zone = {"AC": 1, "AM": 1, "AP": 1, "PA": 1, "RO": 1, "RR": 1, "TO": 1, "AL": 2, "BA": 2, "CE": 2, "MA": 2, "PB": 2, "PE": 2, "PI": 2, "RN": 2, "SE": 2, "DF": 2, "GO": 2, "MT": 2, "MS": 2, "MG": 2, "ES": 2, "RJ": 2, "SP": 4, "PR": 4, "SC": 5, "RS": 6}
        
        z_monthly = z_dengue_monthly.copy()
        records = []
        for _, row in z_monthly.iterrows():
            ym = row["year_month"]
            zone = int(row["climate_zone"])
            inc = row["incidence_rate"]
            cases = row["cases"]
            states_in_zone = [uf for uf, z in state_to_zone.items() if z == zone]
            for uf in states_in_zone:
                records.append({
                    "year_month": ym, "uf": uf, "climate_zone": f"Zone {zone}", "zone_incidence": inc, "zone_cases": cases
                })
        zone_state_df = pd.DataFrame(records).sort_values("year_month")
        
        fig_zone_choropleth = px.choropleth(
            zone_state_df,
            geojson=brazil_geojson,
            locations="uf",
            color="zone_incidence",
            featureidkey="properties.sigla",
            hover_name="uf",
            hover_data={"year_month": True, "climate_zone": True, "zone_cases": ":,.0f", "zone_incidence": ":.2f"},
            animation_frame="year_month",
            color_continuous_scale="Portland",
            range_color=[0, zone_state_df["zone_incidence"].quantile(0.95)],
            title="Brazil Climate Zone Dengue Incidence Forecast (2025-2026)"
        )
        fig_zone_choropleth.update_geos(fitbounds="locations", visible=False)
        fig_zone_choropleth.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar=dict(title="Zone Incidence<br>(per 100k)"))
        
        html_zone_choro_path = "final/outputs_2years/maps/climate_zone_dengue_choropleth.html"
        fig_zone_choropleth.write_html(html_zone_choro_path)
        print(f"Saved Climate Zone Choropleth map to {html_zone_choro_path}")
        
    except Exception as e:
        print("Choropleth generation skipped or failed:", e)

    print("Map generation finished. Interactive maps saved to final/outputs_2years/maps/")

if __name__ == "__main__":
    generate_maps_2years()
