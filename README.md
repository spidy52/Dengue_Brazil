# Spatiotemporal Machine Learning & Climate-Driven Dengue Forecasting in Brazil (2025–2030)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Publication Ready: 600 DPI](https://img.shields.io/badge/Figures-600_DPI_EPS-purple.svg)]()

This repository contains a machine learning framework for spatiotemporal dengue fever epidemic forecasting across **5,561 municipalities** in Brazil, stratified into **6 Macro-Climate Zones**. 

The system leverages multi-source climate reanalysis (temperature, precipitation, relative humidity, pressure, thermal range) combined with historical epidemiological surveillance (2014–2024) to train zone-specific LightGBM regressors, generating **2-year** and **5-year** dengue incidence forecasts (2025–2030).

---

## 🏆 Model Performance & Validation Summary

Evaluated on out-of-time unseen temporal holdout test data (**2023–2024**):

| Aggregation Level | $R^2$ Score | Pearson $r$ | Spearman $\rho$ | MAE | RMSE | Outbreak ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zone-Level Direct Forecast** | **0.7694 – 0.9177** | **0.9890** | **0.9788** | **1.28 – 15.07 /100k** | **4.12 – 23.96 /100k** | **0.9377** |
| **Pooled Municipality Incidence** | **0.8561** | **0.9240** | **0.9110** | **4.12 /100k** | **12.80 /100k** | **0.9377** |

---

## 📊 Key Publication Figures

### 1. High-Resolution Dengue Forecast (2-Year Horizon)
![2-Year Dengue Forecast](figures/dengue_forecast_2years.png)

### 2. High-Resolution Dengue Forecast (5-Year Horizon)
![5-Year Dengue Forecast](figures/dengue_forecast_5years.png)

### 3. Outbreak Detection ROC-AUC Curve (AUC = 0.9377)
![Outbreak ROC Curve](figures/dengue_outbreak_roc_curve.png)

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/spidy52/Dengue_Brazil.git
cd Dengue_Brazil
pip install -r requirements.txt
```

### Running Orchestrator Pipelines

```bash
# Run full pipeline (Climate + Dengue + Forecasts + Plots)
python run.py --all

# Train LightGBM dengue models & generate forecasts
python run.py --train-dengue

# Run 2-year dengue forecast & plots
python run.py --forecast-2years

# Run 5-year dengue forecast & plots
python run.py --forecast-5years
```

---

## 📜 Repository Structure

```
Dengue_Brazil/
├── climate/                  # Climate processing & forecasting modules
├── dengue/                   # LightGBM dengue training & prediction scripts
├── final/                    # Evaluation & validation plot generation scripts
├── figures/                  # 600 DPI publication PNG & EPS forecast images
├── run.py                    # Master CLI orchestrator entry point
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.