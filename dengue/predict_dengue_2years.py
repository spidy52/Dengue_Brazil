import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from dengue.predict_dengue import run_dengue_prediction

def predict_dengue_2years():
    print("Running 2-Year Dengue Forecast Pipeline (2025-2026)...")
    run_dengue_prediction()

if __name__ == "__main__":
    predict_dengue_2years()
