# amiaire/config.py
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DEFAULT_INPUT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_REGRESSION_PARAMS_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "regression_params.json")
DEFAULT_CALIBRATION_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# --- Image Processing Parameters ---
ROI_EXTRACTION_SETTINGS = {
    "gaussian_blur_kernel": (9, 9),
    "adaptive_thresh_block_size": 121,
    "adaptive_thresh_c": 2,
    "morph_open_kernel": (11, 11),
    "morph_close_kernel": (21, 21),
    "contour_area_min_ratio": 0.1, # as ratio of max_image_area
    "contour_area_max_ratio": 0.7, # as ratio of max_image_area
    "contour_similarity_threshold": 0.85,
    "corner_refine_window_size": (5, 5),
    "corner_adjust_margin": 15
}

GRAYSCALE_CONVERSION_PARAMS = {
    "target_size": (1000, 1000)
}

BACKGROUND_IMPROVEMENT_PARAMS = {
    "kernel_size": (21, 21),
    "sigma": 10.0
}

CLAHE_PARAMS = {
    "clip_limit": 0.004, 
    "nbins": 12 
}


RESCALE_INTENSITY_PARAMS = {
    "in_range_percent": (0, 20)
}

SAUVOLA_THRESHOLD_PARAMS = {
    "window_size": 21, 
    "k": 0.18,         
    "invert": True
}

# --- Particle Analysis Parameters ---
DEFAULT_FILTER_PARAMETERS = {
    'min_area': 0.00,
    'max_area': 300,
    'min_solidity': 0.3,
    'max_solidity': 1.0,
    'min_aspect_ratio': 0.00,
    'max_aspect_ratio': 4.0,
    'min_feret': 0.00,
    'max_feret': 50.0 
}

# --- Pollution Level Parameters ---
POLLUTION_CALCULATION_PARAMS = {
    "papersensor_size": (0.06, 0.06), # meters
    # Particle diameter and density should be specific to PM10 or PM2.5
    "PM25": {
        "particle_diameter_microns": 2.5, # micrometers
        "particle_density_g_cm3": 1.65    # g/cm^3
    },
    "PM10": {
        "particle_diameter_microns": 10.0, # micrometers
        "particle_density_g_cm3": 1.65   # g/cm^3 
    }
}

