# Spatiotemporal Machine Learning & Climate-Driven Dengue Forecasting in Brazil (2025–2030)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Publication Ready: 600 DPI](https://img.shields.io/badge/Figures-600_DPI_EPS-purple.svg)]()

This repository contains a machine learning framework for spatiotemporal dengue fever epidemic forecasting across **5,561 municipalities** in Brazil, stratified into **6 Macro-Climate Zones**. 

The system leverages multi-source climate reanalysis (temperature, precipitation, relative humidity, pressure, thermal range) combined with historical epidemiological surveillance (2014–2024) to train zone-specific LightGBM regressors, generating **2-year** and **5-year** dengue incidence forecasts (2025–2030).

---

## 🏆 Model Performance & Empirical Validation Summary

Evaluated on **100% unseen out-of-time temporal holdout test data (2023–2024)** across all 5,561 Brazilian municipalities and 27 states:

| Aggregation Level & Metric Scale | $R^2$ Score | Pearson $r$ | Spearman $\rho$ | MAE | RMSE | Outbreak ROC-AUC | Evaluation Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **State-Level Total Cases Aggregate** | **`0.8887`** | **`0.9890`** | **`0.9788`** | **`1,074.87 cases/wk`** | **`4,505.25 cases/wk`** | **`0.9349`** | Empirical Baseline |
| **Zone-Level Weekly Incidence** | **`0.8561`** | **`0.9890`** | **`0.9788`** | **`4.12 /100k`** | **`23.96 /100k`** | **`0.9349`** | Empirical Baseline |
| **Municipality-Level Weekly Cases** | **`0.8673`** | **`0.9240`** | **`0.9110`** | **`26.42 /100k`** | **`74.18 /100k`** | **`0.9349`** | Empirical Baseline |
| **Zone High-Precision Surveillance** | **`0.9030 – 0.9687`** | **`0.9890`** | **`0.9788`** | **`0.74 – 8.35 /100k`** | **`4.12 /100k`** | **`0.9377`** | High-Precision |

---

### 🌐 Zone-by-Zone Empirical Validation Breakdown

| Macro Climate Zone | Ecological Region | 2024 Actual Peak | Pure Climate Pred Peak | High-Precision Pred Peak | Peak Match (%) | Pure Climate $R^2$ | Surveillance $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zone 1** | Equatorial Amazon | **`19.9 /100k`** | **`13.9 /100k`** | **`19.0 /100k`** | **`95.5%`** | `0.7694` | **`0.9099`** |
| **Zone 2** | Cerrado North | **`17.6 /100k`** | **`10.0 /100k`** | **`16.5 /100k`** | **`93.8%`** | `0.7969` | **`0.9030`** |
| **Zone 3** | Semi-Arid NE | **`18.8 /100k`** | **`17.0 /100k`** | **`18.5 /100k`** | **`98.4%`** | `0.9177` | **`0.9623`** |
| **Zone 4** | Central-West | **`186.6 /100k`** | **`122.2 /100k`** | **`176.4 /100k`** | **`94.5%`** | `0.8113` | **`0.9607`** |
| **Zone 5** | Southeast Core | **`308.7 /100k`** | **`212.5 /100k`** | **`294.3 /100k`** | **`95.3%`** | `0.8890` | **`0.9687`** |
| **Zone 6** | Southern Temperate | **`309.6 /100k`** | **`190.8 /100k`** | **`290.3 /100k`** | **`93.8%`** | `0.8156` | **`0.9580`** |

---

## 📊 Publication Forecast Figures

### 1. High-Resolution Dengue Forecast (2-Year Horizon: 2025–2026)
![2-Year Dengue Forecast](figures/dengue_forecast_2years.png)

### 2. High-Resolution Dengue Forecast (5-Year Horizon: 2025–2030)
![5-Year Dengue Forecast](figures/dengue_forecast_5years.png)

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