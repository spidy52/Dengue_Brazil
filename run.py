"""Unified entry point for running all or specific steps of the climate and dengue forecasting pipeline."""
import os
import sys
import time
import argparse

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_output_directories():
    directories = [
        "climate/models",
        "climate/outputs",
        "dengue/models",
        "dengue/outputs",
        "final/outputs/csv",
        "final/outputs/graphs",
        "final/outputs/maps",
        "final/outputs/metrics",
        "final/outputs/feature_importance",
        "final/outputs_2years/csv",
        "final/outputs_2years/graphs",
        "final/outputs_2years/maps"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_step(step_name, func):
    print(f"\n[{step_name}] Running...")
    t0 = time.time()
    try:
        func()
        print(f"-> {step_name} completed in {time.time()-t0:.1f}s.")
    except Exception as e:
        print(f"ERROR in {step_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Brazil Climate and Dengue Forecasting Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run full pipeline from scratch (Steps 1-6)")
    group.add_argument("--train-dengue", action="store_true", help="Train dengue models, run 5-year forecast & visualizations (Steps 3-6)")
    group.add_argument("--forecast-5years", action="store_true", help="Run 5-year dengue forecast, visualizations & maps (Steps 4-6 only)")
    group.add_argument("--forecast-2years", action="store_true", help="Run 2-year dengue forecast, visualizations & maps (Steps 4-6 only)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    create_output_directories()
    
    if args.all:
        print("=" * 60)
        print("Brazil Climate and Dengue Forecasting - Full Pipeline (Steps 1-6)".center(60))
        print("=" * 60)
        from climate.train_climate import train_climate
        from climate.predict_climate import predict_climate
        from dengue.train_dengue import train_dengue
        from dengue.predict_dengue import predict_dengue
        from final.generate_visualizations import generate_visualizations
        from final.generate_maps import generate_maps
        
        run_step("Step 1: Train Climate Models", train_climate)
        run_step("Step 2: Forecast Climate (2025-2029)", predict_climate)
        run_step("Step 3: Train Dengue Models", train_dengue)
        run_step("Step 4: Forecast Dengue (5-Year)", predict_dengue)
        run_step("Step 5: Visualizations (5-Year)", generate_visualizations)
        run_step("Step 6: Animated Maps (5-Year)", generate_maps)
        
    elif args.train_dengue:
        print("=" * 60)
        print("Brazil Dengue Model Training & 5-Year Forecast (Steps 3-6)".center(60))
        print("=" * 60)
        from dengue.train_dengue import train_dengue
        from dengue.predict_dengue import predict_dengue
        from final.generate_visualizations import generate_visualizations
        from final.generate_maps import generate_maps
        
        run_step("Step 3: Train Dengue Models", train_dengue)
        run_step("Step 4: Forecast Dengue (5-Year)", predict_dengue)
        run_step("Step 5: Visualizations (5-Year)", generate_visualizations)
        run_step("Step 6: Animated Maps (5-Year)", generate_maps)
        
    elif args.forecast_5years:
        print("=" * 60)
        print("Brazil Dengue 5-Year Forecast, Visualizations & Maps".center(60))
        print("=" * 60)
        from dengue.predict_dengue import predict_dengue
        from final.generate_visualizations import generate_visualizations
        from final.generate_maps import generate_maps
        
        run_step("Step 4: Forecast Dengue (5-Year)", predict_dengue)
        run_step("Step 5: Visualizations (5-Year)", generate_visualizations)
        run_step("Step 6: Animated Maps (5-Year)", generate_maps)
        
    elif args.forecast_2years:
        print("=" * 60)
        print("Brazil Dengue 2-Year Forecast, Visualizations & Maps".center(60))
        print("=" * 60)
        from dengue.predict_dengue_2years import predict_dengue_2years
        from final.generate_visualizations_2years import generate_visualizations_2years
        from final.generate_maps_2years import generate_maps_2years
        
        run_step("Step 4: Forecast Dengue (2-Year)", predict_dengue_2years)
        run_step("Step 5: Visualizations (2-Year)", generate_visualizations_2years)
        run_step("Step 6: Animated Maps (2-Year)", generate_maps_2years)
        
    print("\n" + "=" * 60)
    print(f"Pipeline completed in {(time.time()-start_time)/60:.1f} minutes.")
    print("=" * 60)

if __name__ == "__main__":
    main()
