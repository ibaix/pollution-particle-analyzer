# amiaire/main.py
import os
import argparse
from . import roi, preprocessing, analysis, plotting, config

def run_analysis_pipeline(image_path: str, output_base_dir: str, model_type: str):
    
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}")
        return

    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    run_output_dir = os.path.join(output_base_dir, image_filename + "_analysis")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Output will be saved in: {run_output_dir}")

    print("\n--- Step 1: ROI Extraction ---")
    extracted_roi_path = roi.roi_extraction(image_path, output_dir=os.path.join(run_output_dir, "roi_process"))
    if not extracted_roi_path:
        print("ROI extraction failed. Exiting pipeline.")
        return

    print("\n--- Step 2: Image Preprocessing ---")
    preprocessing_output_dir = os.path.join(run_output_dir, "preprocessing_steps")
    binary_mask_path, grayscale_roi_path = preprocessing.run_preprocessing_pipeline(
        extracted_roi_path, preprocessing_output_dir
    )

    print("\n--- Step 3: Particle Analysis ---")
    particle_analysis_output_dir = os.path.join(run_output_dir, "particle_analysis_results")
    analysis_results = analysis.analyze_particles(
        binary_image_path=binary_mask_path,
        original_image_path=grayscale_roi_path,
        output_dir=particle_analysis_output_dir,
        filter_params=config.DEFAULT_FILTER_PARAMETERS
    )
    print(f"Number of particles detected: {analysis_results['num_particles']}")
    print(f"Particle area percentage: {analysis_results['area_percentage']:.2f}%")
    print(f"Total particle area: {analysis_results['total_particle_area_pixels']} pixels")

    print("\n--- Step 4: Pollution Level Calculation ---")
    estimated_concentration = None # Initialize
    classification = "Unknown" # Initialize
    try:
        pollution_info = analysis.calculate_pollution_level(
            analysis_results,
            model_type=model_type,
            regression_models_path=config.DEFAULT_REGRESSION_PARAMS_PATH
        )
        estimated_concentration = pollution_info['estimated_standard_concentration_ug_m3']
        print(f"Estimated {model_type} Concentration: {estimated_concentration:.2f} µg/m³")
        
        print("\n--- Step 5: Pollution Classification ---")
        classification = analysis.classify_pollution_level(estimated_concentration)
        print(f"Pollution Classification: {classification}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error in pollution calculation/classification: {e}")
        print("Please ensure 'scripts/train_correlation_models.py' has been run and "
              f"'{config.DEFAULT_REGRESSION_PARAMS_PATH}' exists and contains the model '{model_type}'.")

    print("\n--- Step 6: Generating Summary Plot ---")
    filtered_overlay_for_plot_path = os.path.join(particle_analysis_output_dir, 'filtered_particles_overlay.png')
    
    # Check if the overlay file actually exists before passing its path
    if not os.path.exists(filtered_overlay_for_plot_path):
        print(f"Warning: Filtered overlay image not found at {filtered_overlay_for_plot_path} for summary plot. Plotting without it.")
        # Pass None if the file doesn't exist, plot_key_processing_stages must handle this.
        filtered_overlay_for_plot_path_or_none = None 
    else:
        filtered_overlay_for_plot_path_or_none = filtered_overlay_for_plot_path

    plotting.plot_key_processing_stages(
        output_dir=run_output_dir,
        original_grayscale_path=grayscale_roi_path,
        binary_mask_path=binary_mask_path,
        filtered_overlay_path=filtered_overlay_for_plot_path_or_none # Pass the potentially None path
    )
    
    summary_text_path = os.path.join(run_output_dir, "analysis_summary.txt")
    with open(summary_text_path, "w") as f:
        f.write(f"Analysis Summary for Image: {image_path}\n")
        f.write("="*30 + "\n")
        f.write(f"Particles Detected: {analysis_results['num_particles']}\n")
        f.write(f"Particle Area Percentage: {analysis_results['area_percentage']:.2f}%\n")
        if estimated_concentration is not None: 
            f.write(f"Estimated {model_type} Concentration: {estimated_concentration:.2f} µg/m³\n")
        f.write(f"Pollution Classification: {classification}\n") # Use initialized or updated value
    print(f"Text summary saved to: {summary_text_path}")

    print("\nPipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amiaire Air Pollution Analysis Pipeline.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_dir", type=str, default=config.DEFAULT_OUTPUT_DIR,
                        help=f"Base directory to save analysis results. Default: {config.DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--model_type", type=str, required=True, choices=["PM10", "PM25"],
                        help="Type of pollution model to use (e.g., PM10, PM25). Must match a key in regression_params.json.")
    
    args = parser.parse_args()
    
    run_analysis_pipeline(args.image_path, args.output_dir, args.model_type)