# Dengue Brazil: Climate-Driven Epidemiological Forecasting

A production-grade, memory-optimised two-stage machine learning pipeline that predicts municipality-level dengue incidence rates across all 27 Brazilian states using LightGBM models driven by climate projections.

---

## 📊 Model Performance & Validation Summary

Evaluated on multi-year unseen test set (2022–2024 out-of-time holdout):

| Aggregation Level | $R^2$ Score | Pearson $r$ | Spearman $\rho$ | MAE | RMSE | Outbreak ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zone-Level Weekly Incidence** | **0.7694 – 0.9177** | **0.9890** | **0.9788** | **1.28 – 15.07 /100k** | **4.12 /100k** | **0.9349** |
| **Pooled Municipality Incidence**| **0.8561** | **0.9240** | **0.9110** | **4.12 /100k** | **12.80 /100k** | **0.9349** |

---

## 📈 Dengue Multi-Year Forecast Curves

### 5-Year Forecast Horizon (2024–2028 Macro-Climate Zones)
![5-Year Combined Dengue Forecast](figures/dengue_forecast_combined_zones_5years.png)

### 2-Year Forecast Horizon (2024–2025 Macro-Climate Zones)
![2-Year Combined Dengue Forecast](figures/dengue_forecast_combined_zones_2years.png)

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
│   ├── train_climate.py        # LightGBM climate model trainer
│   └── predict_climate.py      # Multi-step recursive climate forecast
│
├── dengue/                     # Stage 2: Dengue Model Training & Prediction
│   ├── train_dengue.py         # Dynamic LightGBM dengue trainer
│   ├── predict_dengue.py       # High-precision 5-Year Dengue Forecast (2024-2028)
│   └── predict_dengue_2years.py# 2-Year Dengue Forecast (2024-2025)
│
├── data/                       # Geospatial Metadata
│   └── brazil_states.geojson   # GeoJSON boundaries for 27 Brazilian states
│
├── figures/                    # Forecast Publication Figures (PNG & Vector EPS)
│   ├── dengue_forecast_combined_zones_2years.png / .eps
│   ├── dengue_forecast_combined_zones_5years.png / .eps
│   └── dengue_forecast_zone_1_* to zone_6_*
│
├── final/                      # Master Visualization Scripts
│   ├── generate_visualizations.py
│   └── generate_visualizations_2years.py
│
├── run.py                      # Unified master pipeline orchestrator
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
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