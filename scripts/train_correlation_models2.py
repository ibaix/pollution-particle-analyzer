# scripts/train_correlation_models.py
import numpy as np
import json
import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys
import copy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from main import config

def load_calibration_data(json_file_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Calibration data file not found: {json_file_path}")
    with open(json_file_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if not ('sensor_area_percentage' in df.columns and 'atmotube_concentration' in df.columns):
        raise ValueError("JSON must contain 'sensor_area_percentage' and 'atmotube_concentration' keys for each case.")
    return df['sensor_area_percentage'].values.astype(float), df['atmotube_concentration'].values.astype(float)

def evaluate_loocv(model, X, y):
    """Evaluates a model using Leave-One-Out Cross-Validation and returns LOOCV metrics."""
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        from sklearn.base import clone
        cloned_model = clone(model)
        
        cloned_model.fit(X_train, y_train)
        pred = cloned_model.predict(X_test)[0]
        
        y_true.append(y_test[0])
        y_pred.append(pred)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    loocv_r2 = r2_score(y_true, y_pred)
    
    return loocv_r2, rmse, mae, y_pred

def analyze_poly(paper_sensor_data: np.ndarray, atmotube_data: np.ndarray, degree: int):
    X = paper_sensor_data.reshape(-1, 1)
    y = atmotube_data
    
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    loocv_r2, rmse, mae, cv_y_pred = evaluate_loocv(model, X, y)
    
    model.fit(X, y)
    train_r2 = r2_score(y, model.predict(X))
    
    return {
        'model_type': 'polynomial',
        'desc': f'Poly Degree {degree}',
        'model': model,
        'loocv_r2': loocv_r2,
        'train_r2': train_r2,
        'rmse': rmse,
        'mae': mae,
        'cv_y_pred': cv_y_pred
    }

def analyze_svr(paper_sensor_data: np.ndarray, atmotube_data: np.ndarray, C: float, gamma: str, epsilon: float):
    X = paper_sensor_data.reshape(-1, 1)
    y = atmotube_data
    
    model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon))
    
    loocv_r2, rmse, mae, cv_y_pred = evaluate_loocv(model, X, y)
    
    model.fit(X, y)
    train_r2 = r2_score(y, model.predict(X))
    
    return {
        'model_type': 'svr',
        'desc': f'SVR C={C} gamma={gamma} eps={epsilon}',
        'model': model,
        'loocv_r2': loocv_r2,
        'train_r2': train_r2,
        'rmse': rmse,
        'mae': mae,
        'cv_y_pred': cv_y_pred
    }

def plot_correlation(paper_data: np.ndarray, atmotube_data: np.ndarray, results: dict, model_name: str, output_dir: str, best_flag: bool = False):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(paper_data, atmotube_data, color='blue', alpha=0.6, label='True Data points', edgecolors='k')
    plt.scatter(paper_data, results['cv_y_pred'], color='green', marker='x', s=60, label='LOOCV Predictions')
    
    x_min, x_max = np.min(paper_data), np.max(paper_data)
    x_range = np.linspace(x_min - (x_max - x_min)*0.05, x_max + (x_max - x_min)*0.05, 500)
    
    y_line = results['model'].predict(x_range.reshape(-1, 1))
    
    plt.plot(x_range, y_line, color='red', linewidth=2, label=f"Fit ({results['desc']})")
    
    metrics_text = f"Train R² = {results['train_r2']:.4f}\nLOOCV R² = {results['loocv_r2']:.4f}\nLOOCV RMSE = {results['rmse']:.4f}\nLOOCV MAE = {results['mae']:.4f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
             
    plt.xlabel("Sensor Area Percentage Coverage (%)", fontsize=12)
    plt.ylabel("Atmotube Sensor Concentration (µg/m³)", fontsize=12)
    
    prefix = "[BEST] " if best_flag else ""
    plt.title(f"{prefix}Correlation Model: {model_name} - {results['desc']}", fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs(output_dir, exist_ok=True)
    
    safe_desc = results['desc'].replace('=', '').replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
    filename_prefix = "BEST_" if best_flag else ""
    plot_path = os.path.join(output_dir, f"{filename_prefix}{model_name}_{results['model_type']}_{safe_desc}_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train correlation models.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the data directory containing pm10_training_data.json")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_json_path = config.DEFAULT_REGRESSION_PARAMS_PATH
    plot_dir = os.path.dirname(output_json_path)

    print(f"Using output directory: {plot_dir}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    data_file_path = os.path.join(data_dir, "pm10_training_data.json")
    model_name = "PM10_Clean"
    
    print(f"\n--- Model Evaluation for {model_name} ---")
    try:
        paper_data, atmotube_data = load_calibration_data(data_file_path)
        print(f"Loaded {len(paper_data)} clean points for LOOCV vs Train Evaluation.\n")
        
        all_results = []
        
        for degree in [2, 3]:
            res = analyze_poly(paper_data, atmotube_data, degree=degree)
            all_results.append(res)
            print(f"{res['desc']:<30} | Train R²: {res['train_r2']:>7.4f} | LOOCV R²: {res['loocv_r2']:>7.4f} | RMSE: {res['rmse']:>7.4f}")
            
        C_values = [0.1, 1.0, 10.0, 50.0, 100.0]
        gamma_values = ['scale', 'auto']
        epsilon_values = [0.1, 0.5, 1.0, 2.0]
        
        for C in C_values:
            for gamma in gamma_values:
                for eps in epsilon_values:
                    res = analyze_svr(paper_data, atmotube_data, C=C, gamma=gamma, epsilon=eps)
                    all_results.append(res)
                    print(f"{res['desc']:<30} | Train R²: {res['train_r2']:>7.4f} | LOOCV R²: {res['loocv_r2']:>7.4f} | RMSE: {res['rmse']:>7.4f}")
                    
        # Find the best model based strictly on LOOCV
        best_model_res = max(all_results, key=lambda x: x['loocv_r2'])
        
        print(f"\n========================================================")
        print(f"BEST MODEL OVERALL (By LOOCV): {best_model_res['desc']}")
        print(f"Train R²: {best_model_res['train_r2']:.4f}")
        print(f"LOOCV R²: {best_model_res['loocv_r2']:.4f}")
        print(f"LOOCV RMSE: {best_model_res['rmse']:.4f}")
        print(f"========================================================\n")
        
        plot_correlation(paper_data, atmotube_data, best_model_res, model_name, plot_dir, best_flag=True)
        
        poly2_res = next((r for r in all_results if r['desc'] == 'Poly Degree 2'), None)
        if poly2_res and poly2_res != best_model_res:
             plot_correlation(paper_data, atmotube_data, poly2_res, model_name, plot_dir, best_flag=False)
             
    except Exception as e:
        print(f"Error during execution: {e}")