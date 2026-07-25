# Dengue Brazil: Climate-Driven Epidemiological Forecasting

A production-grade, memory-optimised two-stage machine learning pipeline that predicts municipality-level dengue incidence rates across all 27 Brazilian states using LightGBM models driven by climate projections.

---

## 📊 Model Performance & Dynamic Validation Summary

Evaluated on multi-year unseen test sets (2023–2024 out-of-time holdout):

| Metric Level | $R^2$ Score | Pearson $r$ | Spearman $\rho$ | MAE | Outbreak ROC-AUC | Target Met |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zone-Level Weekly Cases (Macro-Climate Zones)** | **0.8950** | **0.9890** | **0.9788** | 26.42 /100k | **0.9349** | ✅ $> 0.88$ Passed |

---

## 📈 Dengue Forecast Curves (2018–2030)

Each plot displays three seamlessly connected sections:

| Section | Line Style | Period |
|---|---|---|
| **Historical Ground Truth** | Solid blue | 2018 – June 2, 2024 |
| **Validation Buffer** | Dashed green | June 9, 2024 – December 28, 2025 |
| **Forecast Horizon** | Dotted orange ($\pm 1\sigma$ ribbon) | January 4, 2026 – December 28, 2030 |

### Climate Zone 1 - Equatorial Amazon (AC, AM, AP, PA, RO, RR)
![Zone 1 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_1.png)

### Climate Zone 2 - Cerrado North (MA, TO)
![Zone 2 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_2.png)

### Climate Zone 3 - Semi-Arid Northeast (AL, CE, PB, PE, PI, RN, SE)
![Zone 3 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_3.png)

---

## 📊 Regional Validation & 2025 Prediction Summary

| Climate Zone | States Included | Actual 2025 | Predicted 2025 | Error % |
|:---|:---|:---:|:---:|:---:|
| **Zone 1 - Equatorial Amazon** | AC, AM, AP, PA, RO, RR | ~36,800 | 36,594 | <1.0% |
| **Zone 2 - Cerrado North** | MA, TO | ~8,980 | 9,026 | <1.0% |
| **Zone 3 - Semi-Arid NE** | AL, CE, PB, PE, PI, RN, SE | ~64,393 | 63,882 | <1.0% |
| **Zone 4 - Central-West** | BA, DF, GO, MS, MT | ~195,110 | 201,251 | 3.1% |
| **Zone 5 - Southeast Core** | ES, MG, RJ, SP | ~1,132,300 | 1,174,884 | 3.8% |
| **Zone 6 - Southern Temperate** | PR, RS, SC | ~222,167 | 222,513 | 0.2% |
| **National Total** | **All 27 States** | **1,660,186** | **1,708,150** | **2.89%** |

---

## 📁 Project Structure

```
dengue2/
├── climate/                    # Stage 1: Climate Model Training & Prediction
│   ├── train_climate.py        # LightGBM climate model trainer (9 weather targets)
│   └── predict_climate.py      # Multi-step recursive climate forecast (2025-2030)
│
├── dengue/                     # Stage 2: Dengue Model Training & Prediction
│   ├── train_dengue.py         # Dynamic LightGBM dengue trainer (log-target, out-of-time val)
│   ├── predict_dengue.py       # High-precision Dengue Forecast (2025-2030)
│   └── predict_dengue_2years.py# 2-Year Dengue Forecast (2025-2026)
│
├── data/                       # Geospatial & Coordinate Metadata
│   ├── municipios_coords.csv   # Coordinates for 5,570 Brazilian municipalities
│   └── brazil_states.geojson   # GeoJSON boundaries for 27 Brazilian states
│
├── final/                      # Outputs & Visualisations
│   ├── generate_visualizations.py        # 5-Year 600DPI publication plot generator
│   ├── generate_visualizations_2years.py # 2-Year 600DPI publication plot generator
│   │
│   ├── outputs/                # 5-YEAR OUTPUTS (2025-2030)
│   │   ├── graphs/             # Zone forecast curves (600DPI .png + vector .eps)
│   │   ├── metrics/            # Dynamic evaluation metrics (dengue_metrics.csv)
│   │   └── maps/               # Interactive animated choropleth risk maps (.html)
│   │
│   └── outputs_2years/         # 2-YEAR OUTPUTS (2025-2026)
│       ├── graphs/             # 2-Year forecast curves (.eps & .png)
│       └── maps/               # 2-Year interactive risk maps
│
├── final_brazil_dengue.csv     # Raw dataset (2010-2024, 617MB, gitignored)
├── report.tex                  # Formal IEEE / PLOS manuscript draft
├── run.py                      # Unified master pipeline orchestrator
└── requirements.txt            # Python dependencies
```

---

## 🚀 How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run via master orchestrator `run.py`:

| Command | Description |
|---|---|
| `python run.py --all` | Run full pipeline (Climate + Dengue + Plots) |
| `python run.py --train-dengue` | Train LightGBM dengue models & generate forecasts |
| `python run.py --forecast-5years` | Run 5-year dengue forecast & plots |
| `python run.py --forecast-2years` | Run 2-year dengue forecast & plots |