# Dengue Brazil: Climate-Driven Epidemiological Forecasting

A production-grade, memory-optimised end-to-end forecasting pipeline that predicts municipality-level dengue incidence rates across all 27 Brazilian states using LightGBM models driven by climate covariates.

---

## Dengue Forecast Curves

Each plot shows three clearly separated sections:

| Section | Line Style | Period |
|---|---|---|
| **Historical** | Solid blue | 2018 - June 2, 2024 |
| **Validation Buffer** | Dashed green | June 2024 - December 2025 |
| **Forecast** | Dotted orange +/- 1 sigma ribbon | 2026 - 2030 |

### Climate Zone 1 - Equatorial Amazon (AC, AM, AP, PA, RO, RR)
![Zone 1 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_1.png)

### Climate Zone 2 - Cerrado North (MA, TO)
![Zone 2 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_2.png)

### Climate Zone 3 - Semi-Arid Northeast (AL, CE, PB, PE, PI, RN, SE)
![Zone 3 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_3.png)

---

## Zone-Level Validation Summary (2025 Validation Buffer)

| Climate Zone | States | Actual 2025 | Predicted 2025 | Rel. Error |
|:---|:---|:---:|:---:|:---:|
| Zone 1 - Equatorial Amazon | AC, AM, AP, PA, RO, RR | ~36,800 | 36,594 | <1% |
| Zone 2 - Cerrado North | MA, TO | ~8,980 | 9,026 | <1% |
| Zone 3 - Semi-Arid NE | AL, CE, PB, PE, PI, RN, SE | ~64,393 | 63,882 | <1% |
| Zone 4 - Central-West | BA, DF, GO, MS, MT | ~195,110 | 201,251 | 3.1% |
| Zone 5 - Southeast Core | ES, MG, RJ, SP | ~1,132,300 | 1,174,884 | 3.8% |
| Zone 6 - Southern Temperate | PR, RS, SC | ~222,167 | 222,513 | 0.2% |
| **National** | **All 27 States** | **1,660,186** | **1,708,150** | **2.89%** |

---

## Forecast Horizon Peak Incidence per 100k (2026-2030)

| Zone | 2025 Buffer | 2026 | 2027 | 2028 | 2029 | 2030 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Z1 Amazon | 14.0 | 9.2 | ~11 | 16.4 | 5.2 | 8.2 |
| Z2 Cerrado N | 5.5 | 14.4 | ~17 | 19.5 | 10.1 | 4.8 |
| Z3 Semi-Arid NE | 7.8 | 12.8 | ~20 | 33.7 | 10.3 | 18.6 |
| Z4 Central-West | 37.2 | 29.5 | ~33 | 43.2 | 31.5 | 20.0 |
| Z5 SE Core | 83.6 | 17.1 | ~21 | 24.7 | 29.9 | 22.2 |
| Z6 Southern | 60.0 | 30.8 | ~38 | 68.6 | 45.6 | 64.6 |

---

## Project Structure

```
dengue2/
950094729472 climate/                    # Climate Model Training & Prediction
9474   950094729472 train_climate.py        # LightGBM climate model trainer
9474   949294729472 predict_climate.py      # Climate forecast (temp, precip, humidity)
9474
950094729472 dengue/                     # Dengue Model Training & Prediction
9474   950094729472 train_dengue.py         # LightGBM dengue trainer (2018-2024 data)
9474   950094729472 predict_dengue.py       # 5-Year Dengue Forecast (2025-2030)
9474   949294729472 predict_dengue_2years.py# 2-Year Dengue Forecast (2025-2026)
9474
950094729472 data/                       # Geospatial Data
9474   950094729472 municipios_coords.csv   # Coordinates for 5,570 municipalities
9474   949294729472 brazil_states.geojson   # GeoJSON boundary map of 27 states
9474
950094729472 final/                      # Consolidated Outputs & Visualisation
9474   950094729472 generate_visualizations.py        # 5-Year publication plot generator
9474   950094729472 generate_visualizations_2years.py # 2-Year publication plot generator
9474   9474
9474   950094729472 outputs/                # 5-YEAR OUTPUTS (2025-2030)
9474   9474   950094729472 graphs/             # Zone forecast curves (600DPI .png + .eps)
9474   9474   950094729472 metrics/            # Model performance metrics
9474   9474   949294729472 maps/               # Interactive animated choropleth maps (HTML)
9474   9474
9474   949294729472 outputs_2years/         # 2-YEAR OUTPUTS (2025-2026)
9474       950094729472 graphs/             # 2-Year forecast curves (.eps & .png)
9474       949294729472 maps/               # 2-Year interactive maps
9474
950094729472 final_brazil_dengue.csv     # Raw dataset (2010-2024, 617MB, gitignored)
950094729472 run.py                      # Unified master pipeline orchestrator
949294729472 requirements.txt            # Python dependencies
```

---

## How to Run

Install dependencies:
``ash
pip install -r requirements.txt
``

Run via run.py:

| Command | Description |
|---|---|
| `python run.py --all` | Full end-to-end pipeline |
| `python run.py --train-dengue` | Train models + 5-year forecast |
| `python run.py --forecast-5years` | 5-year forecast only |
| `python run.py --forecast-2years` | 2-year forecast only |

---

## Model Details

- **Algorithm**: LightGBM (Gradient Boosted Trees) - one model per climate zone
- **Features**: Weekly epidemiological lags, temperature, precipitation, relative humidity, population density, epiweek seasonality
- **Training Period**: 2010-2023
- **Validation Buffer**: June 2024 - December 2025
- **Forecast Horizon**: 2026-2030 (5 years, weekly resolution)
- **Publication Format**: Times New Roman, 600 DPI PNG + EPS vector, no title, no gridlines
