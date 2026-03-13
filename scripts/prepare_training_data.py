import os
import glob
import pandas as pd
import json
import sys
import argparse

# Ensure amiaire is accessible
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main import roi, preprocessing, analysis, config

def process_image(image_path: str, output_dir: str) -> float:
    """Processes a single image and returns the calculated PM2.5 sensor concentration."""
    print(f"    Processing image: {os.path.basename(image_path)}")
    
    # 1. ROI Extraction
    extracted_roi_path = roi.roi_extraction(image_path, output_dir=os.path.join(output_dir, "roi_process"))
    if not extracted_roi_path:
        print(f"      ROI extraction failed for {image_path}")
        return None

    # 2. Preprocessing
    preprocessing_output_dir = os.path.join(output_dir, "preprocessing_steps")
    binary_mask_path, grayscale_roi_path = preprocessing.run_preprocessing_pipeline(
        extracted_roi_path, preprocessing_output_dir
    )

    # 3. Particle Analysis
    particle_analysis_output_dir = os.path.join(output_dir, "particle_analysis_results")
    analysis_results = analysis.analyze_particles(
        binary_image_path=binary_mask_path,
        original_image_path=grayscale_roi_path,
        output_dir=particle_analysis_output_dir,
        filter_params=config.DEFAULT_FILTER_PARAMETERS
    )

    # 4. Extract Sensor Covarage Metric (using pixel area ratio)
    sensor_area_percentage, _ = analysis.calculate_sensor_cover_metric(analysis_results)
    
    return sensor_area_percentage


def process_case(case_dir: str) -> dict:
    """Processes a single case folder (containing a CSV and images)."""
    print(f"\nProcessing case: {case_dir}")
    
    # 1. Process CSV for Atmotube PM10 baseline
    csv_files = glob.glob(os.path.join(case_dir, "*.csv"))
    if not csv_files:
        print(f"  Warning: No CSV file found in {case_dir}. Skipping case.")
        return None
    
    csv_path = csv_files[0]
    print(f"  Found CSV: {os.path.basename(csv_path)}")
    
    try:
        df = pd.read_csv(csv_path)
        if 'PM10, ug/m3' not in df.columns:
            print(f"  Warning: 'PM10, ug/m3' not in CSV {csv_path}. Skipping case.")
            return None
        
        # Calculate mean PM10 (excluding NaNs)
        df_pm10 = df.dropna(subset=['PM10, ug/m3'])
        if df_pm10.empty:
            print(f"  Warning: No valid PM10 values found in {csv_path}. Skipping.")
            return None
        
        atmotube_mean_pm10 = df_pm10['PM10, ug/m3'].mean()
        print(f"  Atmotube Mean PM10: {atmotube_mean_pm10:.2f}")

        params_to_extract = [
            "VOC, ppm", "AQS", "Temperature, °C", "Humidity, %", 
            "Pressure, mbar", "PM1, ug/m3", "PM2.5, ug/m3", "PM10, ug/m3", 
            "Latitude", "Longitude"
        ]
        
        atmotube_data = {}
        for param in params_to_extract:
            if param in df.columns:
                df_param = df.dropna(subset=[param])
                if not df_param.empty:
                    atmotube_data[param] = df_param[param].mean()
                else:
                    atmotube_data[param] = None
            else:
                atmotube_data[param] = None

    except Exception as e:
        print(f"  Error reading CSV {csv_path}: {e}")
        return None

    # 2. Process all images
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(case_dir, ext)))
    
    if not image_paths:
        print(f"  Warning: No images found in {case_dir}. Skipping case.")
        return None

    print(f"  Found {len(image_paths)} images.")
    sensor_concentrations = []
    
    for img_path in image_paths:
        output_dir = os.path.join(os.path.dirname(img_path), "analysis_output", os.path.basename(img_path))
        perc = process_image(img_path, output_dir)
        if perc is not None:
            sensor_concentrations.append(perc)
            
    if not sensor_concentrations:
        print(f"  Warning: All images failed processing in {case_dir}. Skipping case.")
        return None
        
    avg_area_percentage = sum(sensor_concentrations) / len(sensor_concentrations)
    print(f"  Average Sensor Area Percentage: {avg_area_percentage:.4f}%")
    
    case_path_normalized = os.path.normpath(case_dir)
    data_sep = os.sep + "data" + os.sep
    if data_sep in case_path_normalized:
        short_case_path = data_sep + case_path_normalized.split(data_sep)[-1]
    else:
        short_case_path = case_dir

    result = {
        "case": os.path.basename(case_dir),
        "case_path": short_case_path,
        "sensor_area_percentage": avg_area_percentage
    }
    result.update(atmotube_data)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Prepare training data from images and CSVs.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the data directory containing sequence folders.")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_json_path = os.path.join(data_dir, "training_data.json")
    
    all_data = []

    # Iterate through numbered sequence folders (1, 2, 3...)
    for week_folder in os.listdir(data_dir):
        week_path = os.path.join(data_dir, week_folder)
        if not os.path.isdir(week_path):
            continue
            
        # Iterate through case folders (s1, s2, s3...)
        for case_folder in os.listdir(week_path):
            case_dir = os.path.join(week_path, case_folder)
            if not os.path.isdir(case_dir):
                continue
                
            case_result = process_case(case_dir)
            if case_result:
                all_data.append(case_result)
                
    # Add manually collected PM10 data entries
    manual_data_raw = [
        (0.021, 8.125),
        (0.099, 7.20),
        (0.017, 7.74),
        (0.083, 7.30),
        (0.029, 9.63),
        (0.094, 16.32),
        (0.036, 13.40)
    ]
    
    for i, (area, pm10) in enumerate(manual_data_raw):
        # We add dummy None values for the other atmotube parameters normally extracted
        manual_entry = {
            "case": f"manual_added_pm10_{i}",
            "case_path": f"missing/manual_added_{i}",
            "sensor_area_percentage": area,
            "VOC, ppm": None, "AQS": None, "Temperature, °C": None, 
            "Humidity, %": None, "Pressure, mbar": None, "PM1, ug/m3": None, 
            "PM2.5, ug/m3": None, "PM10, ug/m3": pm10, 
            "Latitude": None, "Longitude": None
        }
        all_data.append(manual_entry)

    print("\n" + "="*40)
    print(f"Processing complete. Compiled {len(all_data) - len(manual_data_raw)} directory data pairs.")
    print(f"Added {len(manual_data_raw)} manual data pairs. Total: {len(all_data)}")
    print("="*40)
    
    # Save the resulting dataset
    with open(output_json_path, 'w') as f:
        json.dump(all_data, f, indent=4)
        
    print(f"Dataset saved to: {output_json_path}")

if __name__ == "__main__":
    main()
