"""
Generate Model Comparison Report
=================================
Creates a comprehensive report showing all 9 trained models' performance.
This serves as evidence for selecting the best model.
"""

import json
import os
import pandas as pd
from pathlib import Path

def generate_model_report():
    """Generate comprehensive model comparison report."""
    
    models_dir = "models"
    report_data = []
    
    # Collect all model metadata
    horizons = [24, 48, 72]
    model_types = ["ridge", "random_forest", "xgboost"]
    
    print("=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)
    print()
    
    for horizon in horizons:
        print(f"\n{'=' * 80}")
        print(f"HORIZON: {horizon}h FORECAST")
        print(f"{'=' * 80}\n")
        
        horizon_models = []
        
        for model_type in model_types:
            # Load metadata
            json_file = os.path.join(models_dir, f"{model_type}_{horizon}h.json")
            joblib_file = os.path.join(models_dir, f"{model_type}_{horizon}h.joblib")
            
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                # Extract metrics (nested under 'metrics' key)
                metrics = metadata.get('metrics', {})
                rmse = metrics.get('rmse', 'N/A')
                mae = metrics.get('mae', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                
                # Model size
                if os.path.exists(joblib_file):
                    size_kb = os.path.getsize(joblib_file) / 1024
                else:
                    size_kb = "N/A"
                
                model_info = {
                    'horizon': horizon,
                    'type': model_type,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'size_kb': size_kb,
                    'metadata': metadata
                }
                
                horizon_models.append(model_info)
                report_data.append(model_info)
                
                print(f"  Model: {model_type.upper()}")
                print(f"    ├─ RMSE: {rmse}")
                print(f"    ├─ MAE:  {mae}")
                print(f"    ├─ R²:   {r2}")
                print(f"    └─ Size: {size_kb} KB" if size_kb != "N/A" else f"    └─ Size: {size_kb}")
                print()
        
        # Find best model for this horizon
        if horizon_models:
            try:
                best = min(horizon_models, key=lambda x: float(x['rmse']) if isinstance(x['rmse'], (int, float)) else float('inf'))
                print(f"  ✓ BEST for {horizon}h: {best['type'].upper()}")
                print(f"    Reason: Lowest RMSE ({best['rmse']})")
                print()
            except (ValueError, TypeError):
                print(f"  ⚠ Could not determine best model (invalid metrics)")
                print()
    
    # Summary comparison table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE - ALL MODELS")
    print(f"{'=' * 80}\n")
    
    df_report = pd.DataFrame(report_data)
    
    # Convert to display format
    display_df = df_report[['horizon', 'type', 'rmse', 'mae', 'r2', 'size_kb']].copy()
    display_df.columns = ['Horizon(h)', 'Model Type', 'RMSE', 'MAE', 'R² Score', 'Size(KB)']
    display_df = display_df.sort_values(['Horizon(h)', 'RMSE'])
    
    print(display_df.to_string(index=False))
    print()
    
    # Save report to file
    report_file = "MODEL_COMPARISON_REPORT.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AQI FORECASTING MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for horizon in horizons:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"HORIZON: {horizon}h FORECAST\n")
            f.write(f"{'=' * 80}\n\n")
            
            horizon_data = [m for m in report_data if m['horizon'] == horizon]
            for model in horizon_data:
                f.write(f"Model: {model['type'].upper()}\n")
                f.write(f"  RMSE: {model['rmse']}\n")
                f.write(f"  MAE:  {model['mae']}\n")
                f.write(f"  R²:   {model['r2']}\n")
                f.write(f"  Size: {model['size_kb']} KB\n\n")
        
        f.write(f"\n{'=' * 80}\n")
        f.write("SUMMARY TABLE\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(display_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        for horizon in horizons:
            horizon_data = [m for m in report_data if m['horizon'] == horizon]
            if horizon_data:
                try:
                    best = min(horizon_data, key=lambda x: float(x['rmse']) if isinstance(x['rmse'], (int, float)) else float('inf'))
                    f.write(f"Best model for {horizon}h: {best['type'].upper()}\n")
                    f.write(f"  Reason: Lowest RMSE ({best['rmse']})\n\n")
                except (ValueError, TypeError):
                    pass
    
    print(f"✓ Report saved to: {report_file}")
    print(f"✓ Use this file as evidence for model selection")
    
    # Save JSON version too
    json_report_file = "MODEL_COMPARISON_REPORT.json"
    with open(json_report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"✓ JSON report saved to: {json_report_file}")
    
    return report_data

if __name__ == "__main__":
    generate_model_report()
