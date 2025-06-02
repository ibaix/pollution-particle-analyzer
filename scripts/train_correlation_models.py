# scripts/train_correlation_models.py
import numpy as np
import json
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from main import config


def load_calibration_data(csv_file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads calibration data from a CSV file."""
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Calibration data file not found: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    if not ('paper_sensor_concentration' in df.columns and 'atmotube_concentration' in df.columns):
        raise ValueError("CSV must contain 'paper_sensor_concentration' and 'atmotube_concentration' columns.")
    
    return df['paper_sensor_concentration'].values.astype(float), df['atmotube_concentration'].values.astype(float)


def analyze_correlation(paper_sensor_data: np.ndarray, atmotube_data: np.ndarray):
    x_reshaped = paper_sensor_data.reshape(-1, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(x_reshaped, atmotube_data)
    slope = lin_reg.coef_[0]
    intercept = lin_reg.intercept_
    y_pred = lin_reg.predict(x_reshaped)
    r_squared = r2_score(atmotube_data, y_pred)
    rmse = np.sqrt(mean_squared_error(atmotube_data, y_pred))
    mae = mean_absolute_error(atmotube_data, y_pred)
    results = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae
    }
    return results

def update_regression_params_json(model_name: str, params: dict, json_path: str):
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    all_params = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                all_params = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: '{json_path}' contains invalid JSON. It will be overwritten.")
            all_params = {}
    all_params[model_name] = {
        'slope': params['slope'],
        'intercept': params['intercept']
    }
    with open(json_path, 'w') as f:
        json.dump(all_params, f, indent=4)
    print(f"Saved/Updated parameters for model '{model_name}' to '{json_path}'")

if __name__ == "__main__":
    models_to_train = {
        "PM10": os.path.join(config.DEFAULT_CALIBRATION_DATA_DIR, "pm10_calibration.csv"),
        "PM25": os.path.join(config.DEFAULT_CALIBRATION_DATA_DIR, "pm25_calibration.csv"),
    }
    output_json_path = config.DEFAULT_REGRESSION_PARAMS_PATH

    for model_name_loop, csv_path_loop in models_to_train.items(): 
        if not os.path.exists(csv_path_loop): 
            print(f"Warning: Calibration file {csv_path_loop} not found. Creating a dummy file.") 
            if model_name_loop == "PM10": 
                dummy_data = {
                    'paper_sensor_concentration': [39635, 53758, 171297, 66211, 30979],
                    'atmotube_concentration': [8.125, 9.63, 16.32, 13.40, 7.74]
                }
            else: 
                dummy_data = {
                    'paper_sensor_concentration': [9969, 13520, 43083, 16652, 7791],
                    'atmotube_concentration': [7.12, 8.44, 14.01, 11.69, 5.38]
                }
            
            dummy_data_dir = os.path.dirname(csv_path_loop)
            if not os.path.exists(dummy_data_dir):
                os.makedirs(dummy_data_dir, exist_ok=True)
            pd.DataFrame(dummy_data).to_csv(csv_path_loop, index=False) 
            print(f"Created dummy data: {csv_path_loop}") 

    print(f"Using regression parameters output file: {output_json_path}")
    if not os.path.exists(os.path.dirname(output_json_path)):
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    all_model_metrics = {}
    for model_name, data_file_path in models_to_train.items():
        print(f"\n--- Training model for {model_name} ---")
        try:
            paper_data, atmotube_data = load_calibration_data(data_file_path)
            print(f"Loaded data: Paper Sensor ({len(paper_data)} points), Atmotube ({len(atmotube_data)} points)")
            training_results = analyze_correlation(paper_data, atmotube_data)
            all_model_metrics[model_name] = training_results
            print(f"Training Results for {model_name}: {training_results}")
            update_regression_params_json(
                model_name=model_name,
                params={'slope': training_results['slope'], 'intercept': training_results['intercept']},
                json_path=output_json_path
            )
        except FileNotFoundError as e:
            print(f"Error for {model_name}: {e}. Skipping this model.")
        except ValueError as e:
            print(f"Data Error for {model_name}: {e}. Skipping this model.")
        except Exception as e:
            print(f"An unexpected error occurred for {model_name}: {e}. Skipping this model.")

    print("\n--- All Model Metrics ---")
    for model_name_metrics, metrics in all_model_metrics.items(): 
        print(f"{model_name_metrics}: R-squared={metrics['r_squared']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")