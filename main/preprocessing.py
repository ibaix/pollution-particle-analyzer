# amiaire/preprocessing.py
import cv2
import numpy as np

from skimage import io, exposure, filters 
from skimage.util import img_as_ubyte 
# from skimage import img_as_float 
import os
from . import config

def convert_to_grayscale_8bit(image_path: str, output_path: str, 
                              target_size: tuple[int, int] = config.GRAYSCALE_CONVERSION_PARAMS["target_size"]): 
    
    if not (isinstance(target_size, tuple) and len(target_size) == 2 and all(isinstance(x, int) and x > 0 for x in target_size)):
        raise ValueError("target_size must be a tuple of two positive integers.")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None: raise FileNotFoundError(f"Image not found: {image_path}")
    if image.ndim == 3:
        if image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2: grayscale_image = image
    else: raise ValueError("Unsupported image format!")
    resized_image = cv2.resize(grayscale_image, target_size, interpolation=cv2.INTER_CUBIC)
    if not cv2.imwrite(output_path, resized_image): raise IOError(f"Failed to save to {output_path}")
    print(f"Grayscale image saved to {output_path}")


def improve_background(image_path: str, output_path: str, 
                       kernel_size: tuple[int, int] = config.BACKGROUND_IMPROVEMENT_PARAMS["kernel_size"], 
                       sigma: float = config.BACKGROUND_IMPROVEMENT_PARAMS["sigma"]):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: raise FileNotFoundError(f"Image not found: {image_path}")
    
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma)
    # Ensure float32 for subtraction to prevent underflow with uint8
    result_image_float = cv2.subtract(image.astype(np.float32), blurred_image.astype(np.float32))
    
    normalized_image = cv2.normalize(src=result_image_float, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # type: ignore
    result_image_uint8 = normalized_image.astype(np.uint8)

    if not cv2.imwrite(output_path, result_image_uint8): raise IOError(f"Failed to save to {output_path}")
    print(f"Background improved image saved to {output_path}")


def clahe_skimage(image_path: str, output_path: str, 
                  clip_limit: float = config.CLAHE_PARAMS["clip_limit"], 
                  nbins: int = config.CLAHE_PARAMS["nbins"]):
    
    image = io.imread(image_path, as_gray=True)
    if image is None: raise FileNotFoundError(f"Image not found: {image_path}")
    clahe_image = exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=nbins)
    clahe_uint8 = img_as_ubyte(clahe_image)
    if not cv2.imwrite(output_path, clahe_uint8): raise IOError(f"Failed to save to {output_path}")
    print(f"CLAHE image saved to {output_path}")
    return clahe_uint8


def rescale_intensity_skimage(image_path: str, output_path: str, 
                              in_range_percent: tuple[float, float] = config.RESCALE_INTENSITY_PARAMS["in_range_percent"]): 
    image = io.imread(image_path, as_gray=True)
    if image is None: raise FileNotFoundError(f"Image not found: {image_path}")
    
    p_low, p_high = np.percentile(image, in_range_percent)
    
    rescaled = exposure.rescale_intensity(image, in_range=(p_low, p_high)) # type: ignore[reportArgumentType]
    rescaled_uint8 = img_as_ubyte(rescaled)
    if not cv2.imwrite(output_path, rescaled_uint8): raise IOError(f"Failed to save to {output_path}")
    print(f"Rescaled intensity image saved to {output_path}")
    return rescaled_uint8


def apply_sauvola_threshold(image_path: str, output_path: str, 
                            window_size: int = config.SAUVOLA_THRESHOLD_PARAMS["window_size"], 
                            k: float = config.SAUVOLA_THRESHOLD_PARAMS["k"], 
                            invert: bool = config.SAUVOLA_THRESHOLD_PARAMS["invert"]):
    image = io.imread(image_path, as_gray=True)
    if image is None: raise FileNotFoundError(f"Image not found: {image_path}")
    thresh_sauvola = filters.threshold_sauvola(image, window_size=window_size, k=k)
    binary_image = image > thresh_sauvola 
    binary_uint8 = img_as_ubyte(binary_image) 
    if invert:
        binary_uint8 = cv2.bitwise_not(binary_uint8)
    if not cv2.imwrite(output_path, binary_uint8): raise IOError(f"Failed to save to {output_path}")
    print(f"Sauvola thresholded image saved to {output_path}")
    return binary_uint8

def run_preprocessing_pipeline(input_image_path: str, output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    gray_output = os.path.join(output_dir, '01_grayscale.jpg')
    convert_to_grayscale_8bit(input_image_path, gray_output)

    bg_output = os.path.join(output_dir, '02_background_improved.jpg')
    improve_background(gray_output, bg_output)

    rescale_output = os.path.join(output_dir, '03_rescale_intensity.jpg')
    rescale_intensity_skimage(bg_output, rescale_output)

    clahe_output = os.path.join(output_dir, '04_clahe_result.jpg')
    clahe_skimage(rescale_output, clahe_output)
    
    binary_mask_output = os.path.join(output_dir, '05_binary_mask.png')
    apply_sauvola_threshold(clahe_output, binary_mask_output)
    
    print("Preprocessing pipeline completed.")
    return binary_mask_output, gray_output