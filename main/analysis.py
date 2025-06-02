# amiaire/analysis.py
import cv2
import numpy as np
from skimage import measure
import os
import json
from . import config # Use relative import


def analyze_particles(binary_image_path: str, 
                      original_image_path: str | None = None, 
                      output_dir: str | None = None, 
                      filter_params: dict | None = None):
    """
    Analyzes particles in a binary image, filters them, and optionally saves overlay images.
    Saves overlays and filtered_mask to output_dir if provided.
    """
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    if binary_image is None:
        raise FileNotFoundError(f"Binary image not found: {binary_image_path}")

    if filter_params is None:
        filter_params = config.DEFAULT_FILTER_PARAMETERS
    
    
    original_img = None
    if original_image_path:
        original_img = cv2.imread(original_image_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
       
        if original_img is not None: 
            overlay_all_path = os.path.join(output_dir, 'debug_all_particles_overlay.png')
            overlay_image_all = original_img.copy() # Use a different variable name for clarity
             


    binary_bool = binary_image > 0 # Convert to boolean mask
    label_image = measure.label(binary_bool)
    regions = measure.regionprops(label_image)
    
    
    if original_img is not None and output_dir:
        overlay_all_path = os.path.join(output_dir, 'debug_all_particles_overlay.png')
        overlay_image_all_debug = original_img.copy()
        for region_debug in regions: # Iterate over all initially found regions
            minr, minc, maxr, maxc = region_debug.bbox
            cv2.rectangle(overlay_image_all_debug, (minc, minr), (maxc, maxr), (255, 0, 0), 1) # Blue for all
        cv2.imwrite(overlay_all_path, overlay_image_all_debug)


    filtered_regions = []
    for region in regions:
        area = region.area
        solidity = region.solidity
        aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else region.major_axis_length
        if aspect_ratio == 0 : aspect_ratio = 1

        feret_diameter = getattr(region, 'feret_diameter_max', region.equivalent_diameter)

       
        if not (filter_params.get('min_area',0) <= area <= filter_params.get('max_area',float('inf'))): continue
        if not (filter_params.get('min_solidity',0) <= solidity <= filter_params.get('max_solidity',1.0)): continue
        if not (filter_params.get('min_aspect_ratio',0) <= aspect_ratio <= filter_params.get('max_aspect_ratio',float('inf'))): continue
        if not (filter_params.get('min_feret',0) <= feret_diameter <= filter_params.get('max_feret',float('inf'))): continue
        filtered_regions.append(region)

    filtered_mask = np.zeros(label_image.shape, dtype=np.uint8) # type: ignore # label_image.shape is a tuple of int
    for region in filtered_regions:
        coords = region.coords
        filtered_mask[coords[:,0], coords[:,1]] = 255
    
    if output_dir:
        filtered_mask_path = os.path.join(output_dir, 'filtered_binary_mask.png')
        cv2.imwrite(filtered_mask_path, filtered_mask)

       
        if original_img is not None: 
            overlay_filtered_path = os.path.join(output_dir, 'filtered_particles_overlay.png')
            overlay_filtered_image = original_img.copy()
            for region in filtered_regions: # Iterate over filtered regions
                minr, minc, maxr, maxc = region.bbox
                cv2.rectangle(overlay_filtered_image, (minc, minr), (maxc, maxr), (0,255,0), 1) # Green for filtered
            cv2.imwrite(overlay_filtered_path, overlay_filtered_image)
            
    # The Pylance errors about filtered_mask.shape are likely due to complex type inference.
    # The code is standard NumPy. 
    image_area_pixels = filtered_mask.shape[0] * filtered_mask.shape[1]
    particle_area_pixels = np.count_nonzero(filtered_mask)
    area_percentage = (particle_area_pixels / image_area_pixels) * 100 if image_area_pixels > 0 else 0

    return {
        'num_particles': len(filtered_regions),
        'total_particle_area_pixels': particle_area_pixels,
        'image_area_pixels': image_area_pixels,
        'area_percentage': area_percentage,
        'filtered_regions': filtered_regions
    }


def calculate_pollution_level(analysis_results: dict, model_type: str, regression_models_path: str = config.DEFAULT_REGRESSION_PARAMS_PATH):
    """
    Calculates pollution concentration based on particle analysis and a regression model.
    """
    if not os.path.exists(regression_models_path):
        raise FileNotFoundError(f"Regression models file not found: {regression_models_path}. Run train_correlation_models.py.")
    with open(regression_models_path, 'r') as f:
        all_models_params = json.load(f)
    
    if model_type not in all_models_params:
        raise ValueError(f"Model type '{model_type}' not found in regression parameters. Available: {list(all_models_params.keys())}")
    model_params = all_models_params[model_type]
    slope = model_params['slope']
    intercept = model_params['intercept']

    calc_params = config.POLLUTION_CALCULATION_PARAMS
    if model_type not in calc_params:
        raise ValueError(f"Pollution calculation parameters for '{model_type}' not defined in config.")
    
    particle_dia_microns = calc_params[model_type]['particle_diameter_microns']
    particle_density_g_cm3 = calc_params[model_type]['particle_density_g_cm3']

    particle_diameter_m = particle_dia_microns * 1e-6
    particle_density_kg_m3 = particle_density_g_cm3 * 1e12

    sensor_width_m, sensor_height_m = calc_params['papersensor_size']
    area_sensor_m2 = sensor_width_m * sensor_height_m
    
    volume_sensor_m3 = sensor_width_m * sensor_height_m * sensor_width_m 

    area_particle_m2 = np.pi * (particle_diameter_m / 2)**2
    volume_particle_m3 = (4/3) * np.pi * (particle_diameter_m / 2)**3

    num_particles_on_sensor = (area_sensor_m2 * (analysis_results['area_percentage'] / 100)) / area_particle_m2
    
    mass_total_particles_kg = num_particles_on_sensor * volume_particle_m3 * particle_density_kg_m3
    concentration_sensor_kg_m3 = mass_total_particles_kg / volume_sensor_m3 if volume_sensor_m3 > 0 else 0
    
    concentration_sensor_ug_m3 = concentration_sensor_kg_m3 * 1e9

    estimated_standard_concentration_ug_m3 = intercept + slope * concentration_sensor_ug_m3

    return {
        'calculated_num_particles': num_particles_on_sensor,
        'sensor_paper_concentration_ug_m3': concentration_sensor_ug_m3,
        'estimated_standard_concentration_ug_m3': estimated_standard_concentration_ug_m3,
        'model_type_used': model_type
    }

def classify_pollution_level(concentration_ug_m3: float):
    """Classifies pollution level based on concentration in µg/m³."""
    if concentration_ug_m3 <= 10: return "Level 1 (Good): 0-10 µg/m³"
    elif 10 < concentration_ug_m3 <= 20: return "Level 2 (Moderate): 11-20 µg/m³"
    elif 20 < concentration_ug_m3 < 50: return "Level 3 (Unhealthy for Sensitive): 21-49 µg/m³"
    elif 50 <= concentration_ug_m3 < 100: return "Level 4 (Unhealthy): 50-99 µg/m³"
    elif 100 <= concentration_ug_m3 < 150: return "Level 5 (Very Unhealthy): 100-149 µg/m³"
    else: return "Level 6 (Hazardous): 150+ µg/m³"