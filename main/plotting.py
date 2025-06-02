# amiaire/plotting.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_key_processing_stages(output_dir: str, 
                               original_grayscale_path: str, 
                               binary_mask_path: str, 
                               filtered_overlay_path: str | None): # Allow None
    """
    Plots the original grayscale image, the final binary mask, and the
    filtered particles overlay.
    """
    original_image = cv2.imread(original_grayscale_path, cv2.IMREAD_GRAYSCALE)
    threshold_image = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
    
    filtered_overlay_bgr = None
    if filtered_overlay_path and os.path.exists(filtered_overlay_path): # Check if path is not None and exists
        filtered_overlay_bgr = cv2.imread(filtered_overlay_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    
    if original_image is not None:
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('1. Grayscale ROI')
    else:
        axes[0].text(0.5, 0.5, 'Grayscale Not Found', ha='center', va='center')
        axes[0].set_title('1. Grayscale ROI')
    axes[0].axis('off')

    if threshold_image is not None:
        axes[1].imshow(threshold_image, cmap='gray')
        axes[1].set_title('2. Binary Mask (Particles)')
    else:
        axes[1].text(0.5, 0.5, 'Binary Mask Not Found', ha='center', va='center')
        axes[1].set_title('2. Binary Mask (Particles)')
    axes[1].axis('off')

    # Filtered Overlay - handles if filtered_overlay_bgr is None
    if filtered_overlay_bgr is not None:
        filtered_overlay_rgb = cv2.cvtColor(filtered_overlay_bgr, cv2.COLOR_BGR2RGB)
        axes[2].imshow(filtered_overlay_rgb)
        axes[2].set_title('3. Filtered Particles Overlay')
    else:
        axes[2].text(0.5, 0.5, 'Overlay Not Found or Not Provided', ha='center', va='center')
        axes[2].set_title('3. Filtered Particles Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, 'summary_processing_results.png')
    plt.savefig(summary_plot_path)
    print(f"Summary plot saved to {summary_plot_path}")
    # plt.show()