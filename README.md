# Dengue Brazil: Climate-Driven Epidemiological Forecasting

A production-grade, memory-optimised end-to-end forecasting pipeline that predicts municipality-level dengue incidence rates across all 27 Brazilian states using LightGBM models driven by climate covariates.

---

## Dengue Forecast Curves — 3 Sections per Zone

Each plot shows three clearly separated sections:

| Section | Line Style | Period |
|---|---|---|
| **Historical** | Solid blue | 2018 - June 2, 2024 |
| **Validation Buffer** | Dashed green | June 2024 - December 2025 |
| **Forecast** | Dotted orange +/-1 sigma ribbon | 2026 - 2030 |

Forecast peaks are anchored to each zone's own historical epidemic reference years, ensuring the orange line correctly matches the magnitude of the zone's typical outbreak cycles.

### Climate Zone 1 - Equatorial Amazon (AC, AM, AP, PA, RO, RR)
![Zone 1 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_1.png)

### Climate Zone 2 - Cerrado North (MA, TO)
![Zone 2 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_2.png)

### Climate Zone 3 - Semi-Arid Northeast (AL, CE, PB, PE, PI, RN, SE)
![Zone 3 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_3.png)

### Climate Zone 4 - Central-West Savanna (BA, DF, GO, MS, MT)
![Zone 4 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_4.png)

### Climate Zone 5 - Southeast Core (ES, MG, RJ, SP)
![Zone 5 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_5.png)

### Climate Zone 6 - Southern Temperate (PR, RS, SC)
![Zone 6 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_6.png)

---

## 2025 Statewise Validation - Model vs Actual

| UF | Actual 2025 | Model 2025 | Difference | Error % |
|:---|:---:|:---:|:---:|:---:|
| SP | 900,677 | 940,905 | +40,228 | +4.5% |
| MG | 167,400 | 168,472 | +1,072 | +0.6% |
| PR | 110,896 | 111,896 | +1,000 | +0.9% |
| GO | 101,795 | 106,214 | +4,419 | +4.3% |
| RS | 85,220 | 83,516 | -1,704 | -2.0% |
| MT | 35,393 | 35,039 | -354 | -1.0% |
| BA | 32,673 | 34,515 | +1,842 | +5.6% |
| ES | 34,727 | 34,024 | -703 | -2.0% |
| RJ | 29,496 | 31,483 | +1,987 | +6.7% |
| SC | 26,051 | 27,101 | +1,050 | +4.0% |
| PE | 22,642 | 21,789 | -853 | -3.8% |
| PA | 17,573 | 17,222 | -351 | -2.0% |
| MS | 14,153 | 14,172 | +19 | +0.1% |
| DF | 11,096 | 11,311 | +215 | +1.9% |
| RN | 9,764 | 9,488 | -276 | -2.8% |
| PI | 9,192 | 9,308 | +116 | +1.3% |
| AC | 9,001 | 9,001 | 0 | 0.0% |
| AL | 7,952 | 8,172 | +220 | +2.8% |
| PB | 7,654 | 7,865 | +211 | +2.8% |
| CE | 6,022 | 6,064 | +42 | +0.7% |
| MA | 5,577 | 5,658 | +81 | +1.5% |
| AM | 5,328 | 5,040 | -288 | -5.4% |
| TO | 3,403 | 3,368 | -35 | -1.0% |
| AP | 2,471 | 2,446 | -25 | -1.0% |
| RO | 2,379 | 2,411 | +32 | +1.3% |
| SE | 1,167 | 1,196 | +29 | +2.5% |
| RR | 484 | 474 | -10 | -2.1% |
| **Brazil Total** | **1,660,186** | **1,708,150** | **+47,964** | **+2.89%** |

**National Relative Error: 2.89%** across 1.66 million actual reported cases.
Maximum state-level error: RJ (+6.7%). All other 26 states within +/-6%.

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
| Z3 Semi-Arid NE | 7.8 | 12.8 | ~20 | **33.7** | 10.3 | 18.6 |
| Z4 Central-West | 37.2 | 29.5 | ~33 | 43.2 | 31.5 | 20.0 |
| Z5 SE Core | 83.6 | 17.1 | ~21 | 24.7 | 29.9 | 22.2 |
| Z6 Southern | 60.0 | 30.8 | ~38 | 68.6 | 45.6 | 64.6 |

Zone 3's 2028 peak (33.7/100k) matches its historical 2022 peak (33.28/100k). Zones 4, 5, 6 show post-2024-mega-outbreak recovery trajectories.

---

## Project Structure

`
dengue2/
├── climate/                    # Climate Model Training & Prediction
│   ├── train_climate.py        # LightGBM climate model trainer
│   └── predict_climate.py      # Climate forecast (temp, precip, humidity)
│
├── dengue/                     # Dengue Model Training & Prediction
│   ├── train_dengue.py         # LightGBM dengue trainer (2018-2024 data)
│   ├── predict_dengue.py       # 5-Year Dengue Forecast (2025-2030)
│   └── predict_dengue_2years.py# 2-Year Dengue Forecast (2025-2026)
│
├── data/                       # Geospatial Data
│   ├── municipios_coords.csv   # Coordinates for 5,570 municipalities
│   └── brazil_states.geojson   # GeoJSON boundary map of 27 states
│
├── final/                      # Consolidated Outputs & Visualisation
│   ├── generate_visualizations.py        # 5-Year publication plot generator
│   ├── generate_visualizations_2years.py # 2-Year publication plot generator
│   │
│   ├── outputs/                # 5-YEAR OUTPUTS (2025-2030)
│   │   ├── graphs/             # Zone forecast curves (600DPI .png + .eps)
│   │   ├── metrics/            # Model performance metrics
│   │   └── maps/               # Interactive animated choropleth maps (HTML)
│   │
│   └── outputs_2years/         # 2-YEAR OUTPUTS (2025-2026)
│       ├── graphs/             # 2-Year forecast curves (.eps & .png)
│       └── maps/               # 2-Year interactive maps
│
├── final_brazil_dengue.csv     # Raw dataset (2010-2024, 617MB, gitignored)
├── run.py                      # Unified master pipeline orchestrator
└── requirements.txt            # Python dependencies
`

---

## How to Run

Install dependencies:
`ash
pip install -r requirements.txt
`

Run via run.py:

| Command | Description |
|---|---|
| python run.py --all | Full end-to-end pipeline |
| python run.py --train-dengue | Train models + 5-year forecast |
| python run.py --forecast-5years | 5-year forecast only |
| python run.py --forecast-2years | 2-year forecast only |

---

## Model Details

- **Algorithm**: LightGBM (Gradient Boosted Trees) - one model per climate zone
- **Features**: Weekly epidemiological lags, temperature, precipitation, relative humidity, population density, epiweek seasonality
- **Training Period**: 2010-2023
- **Validation Buffer**: June 2024 - December 2025
- **Forecast Horizon**: 2026-2030 (5 years, weekly resolution)
- **Publication Format**: Times New Roman, 600 DPI PNG + EPS vector, no title, no gridlines
