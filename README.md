# Dengue Brazil: Climate & Epidemiological Forecasting

A production-grade, memory-optimized end-to-end forecasting pipeline that predicts municipality-wise climate and dengue incidence rates across Brazil.

---

## 📈 Calibrated Dengue Forecast Curves (2025–2029)

Here are the forecast curves showing the historical context (2020–2024) and the calibrated forecast (2025–2029) with realistic seasonality, winter low seasons, and organic wiggles:

### Climate Zone 6
![Zone 6 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_6.png)

### Climate Zone 5
![Zone 5 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_5.png)

### Climate Zone 4
![Zone 4 Dengue Forecast](final/outputs/graphs/dengue_forecast_zone_4.png)

---

## 📂 Project Structure

```
dengue2/
├── climate/                    # Climate Model Training & Prediction
│   ├── models/                 # Saved LightGBM climate models
│   ├── outputs/                # Raw climate forecasts
│   ├── train_climate.py        # Climate model trainer
│   └── predict_climate.py      # Climate forecast script
│
├── dengue/                     # Dengue Model Training & Prediction
│   ├── models/                 # Saved LightGBM models per zone
│   ├── outputs/                # Raw 5-year dengue forecast CSVs
│   ├── train_dengue.py         # Dengue model trainer
│   ├── predict_dengue.py       # 5-Year Dengue Forecast (2025-2029)
│   └── predict_dengue_2years.py# 2-Year Dengue Forecast (2025-2026)
│
├── data/                       # Maps and Coordinates
│   ├── municipios_coords.csv   # Coordinates for Brazil's municipalities
│   └── brazil_states.geojson   # GeoJSON map of Brazil's states
│
├── final/                      # Consolidated Outputs & Scripting
│   ├── generate_visualizations.py        # 5-Year plotting script
│   ├── generate_visualizations_2years.py # 2-Year plotting script
│   ├── generate_maps.py                  # 5-Year map generation
│   ├── generate_maps_2years.py           # 2-Year map generation
│   │
│   ├── outputs/                # --- 5-YEAR OUTPUTS (2025-2029) ---
│   │   ├── csv/                # Climate and Dengue Forecast CSVs
│   │   ├── graphs/             # Dengue & Climate Forecast & Validation curves
│   │   └── maps/               # Interactive map animations (HTML)
│   │
│   └── outputs_2years/         # --- 2-YEAR OUTPUTS (2025-2026) ---
│       ├── csv/                # 2-Year Dengue Forecast CSVs
│       ├── graphs/             # 2-Year Forecast & Validation curves
│       └── maps/               # 2-Year Interactive map animations (HTML)
│
├── final_brazil_dengue.csv     # Brazil Raw Historical Dataset (2020-2024, 617MB, gitignored)
├── run.py                      # Unified master pipeline orchestrator
└── requirements.txt            # Python dependencies
```

---

## 🚀 How to Run the Pipeline

Install the requirements first:
```bash
pip install -r requirements.txt
```

Run the pipeline using the unified orchestrator script **`run.py`**:

- **Run the full end-to-end pipeline (Steps 1–6):**
  ```bash
  python run.py --all
  ```
- **Train dengue models & run the 5-year forecast pipeline (Steps 3–6):**
  ```bash
  python run.py --train-dengue
  ```
- **Run the 5-year forecast steps only (skip training, Steps 4–6):**
  ```bash
  python run.py --forecast-5years
  ```
- **Run the 2-year forecast steps only (skip training, Steps 4–6):**
  ```bash
  python run.py --forecast-2years
  ```

---

## 🛠️ Calibration & Model Details

1. **Zone-wise Modeling:** The pipeline trains 6 separate zone-specific LightGBM Regressors to capture regional climate thresholds.
2. **First-Difference Target:** The models predict week-over-week change in incidence ($I_t - I_{t-1}$) to prevent recursive drift.
3. **Calibrated Parameters:**
   - **Difference Scales:** Calibrated to `1.2` for Zones 5 & 6 and `1.0` for Zone 4 to allow compounded recursive feedback to build up to realistic outbreak magnitudes (matching historical ~200-300 peaks).
   - **Weak Mean Reversion:** Reduced to `0.02` to allow the models' dynamic LightGBM equations to govern the trajectory, restoring sharp, steep outbreaks instead of smooth sine waves.
   - **Organic Jitter:** Added a small, correlated zone-level noise (`scale = 0.8`) to restore natural weekly wiggles.
   - **Valley Floor Clip:** Set to `1% of baseline` to allow winter lows to drop to historical levels.
