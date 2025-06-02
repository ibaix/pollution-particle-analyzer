# amiaire/roi.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from . import config # Use relative import for config


def _order_points(pts):
   
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _warp_perspective_roi(image, corners):
    
    corners = np.array(corners, dtype="float32").reshape(-1, 2)
    if corners.shape[0] != 4:
        raise ValueError("Expected exactly 4 corner points.")
    rect = _order_points(corners)
    (tl, tr, br, bl) = rect
    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    maxWidth = int(max(width_bottom, width_top))
    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    maxHeight = int(max(height_right, height_left))
    dst = np.array([
        [0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
    return warped



def _refine_corners(gray_image, corners):
    
    corners = np.array(corners, dtype=np.float32)
    win_size = config.ROI_EXTRACTION_SETTINGS["corner_refine_window_size"]
    zero_zone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners_reshaped = corners.reshape(-1, 1, 2)
    refined = cv2.cornerSubPix(gray_image, corners_reshaped, win_size, zero_zone, criteria)
    return refined.reshape(-1, 2)


def _adjust_corners(corners, margin=20):
    
    centroid = np.mean(corners, axis=0)
    distances = np.linalg.norm(corners - centroid, axis=1, keepdims=True)
    if np.any(distances == 0): return corners
    adjusted = corners + margin * (centroid - corners) / distances
    return adjusted


def roi_extraction(image_path: str, output_dir: str) -> str | None:
    """
    Extracts the region of interest (ROI) defined by a large black square
    from an input image and saves intermediate and final images to output_dir.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save processed images (binary, ROI figure, extracted ROI).

    Returns:
        str | None: Path to the extracted ROI image if successful, else None.
    """
    os.makedirs(output_dir, exist_ok=True) 

    settings = config.ROI_EXTRACTION_SETTINGS

    color_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if color_image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, settings["gaussian_blur_kernel"], 0)

    binary = cv2.adaptiveThreshold(
        gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, settings["adaptive_thresh_block_size"], settings["adaptive_thresh_c"]
    )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, settings["morph_open_kernel"])
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, settings["morph_close_kernel"])
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    binary_output_path = os.path.join(output_dir, 'roi_binary_debug.jpg')
    cv2.imwrite(binary_output_path, binary)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    max_image_area = h * w

    candidate_contours = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if (settings["contour_area_min_ratio"] * max_image_area < area < 
                    settings["contour_area_max_ratio"] * max_image_area):
                candidate_contours.append(approx)

    candidate_contours.sort(key=cv2.contourArea, reverse=True)

    inner_contour = None
    if len(candidate_contours) >= 2:
        chosen_pair_found = False
        for i in range(len(candidate_contours) - 1):
            area1 = cv2.contourArea(candidate_contours[i])
            area2 = cv2.contourArea(candidate_contours[i + 1])
            ratio = area2 / area1 if area1 > 0 else 0
            if ratio >= settings["contour_similarity_threshold"]:
                inner_contour = candidate_contours[i + 1]
                chosen_pair_found = True
                break
        if not chosen_pair_found:
            inner_contour = candidate_contours[0]
    elif len(candidate_contours) == 1:
        inner_contour = candidate_contours[0]

    extracted_roi_path = None
    roi_image = None
    if inner_contour is not None:
        corners = np.squeeze(inner_contour, axis=1).astype(np.float32)
        refined_corners = _refine_corners(gray_blurred, corners)
        adjusted_corners = _adjust_corners(refined_corners, margin=settings["corner_adjust_margin"])
        roi_image = _warp_perspective_roi(color_image, adjusted_corners)
        
        extracted_roi_path = os.path.join(output_dir, "extracted_roi.jpg")
        cv2.imwrite(extracted_roi_path, roi_image)
        print(f"Extracted ROI image saved to {extracted_roi_path}")
    else:
        print("No valid ROI contour found.")

    # Visualization (optional, can be controlled by a flag)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    color_image_copy = color_image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255)] 
    for i, cnt_candidate in enumerate(candidate_contours[:3]):
        cv2.drawContours(color_image_copy, [cnt_candidate], -1, colors[i], 3) 
    if inner_contour is not None:
         cv2.drawContours(color_image_copy, [inner_contour], -1, (0, 255, 0), 5) 
         plt.title("Detected Inner Square (Green) & Candidates")
    else:
        plt.title("No Valid Square Found / Candidates")
    plt.imshow(cv2.cvtColor(color_image_copy, cv2.COLOR_BGR2RGB))

    if roi_image is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        plt.title("Extracted ROI")
    else:
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, 'No ROI Extracted', horizontalalignment='center', verticalalignment='center')
        plt.title("Extracted ROI")

    plt.tight_layout()
    roi_figure_path = os.path.join(output_dir, "roi_detection_figure.png")
    plt.savefig(roi_figure_path)
    print(f"ROI detection figure saved to {roi_figure_path}")
    # plt.show() # Comment out if running in a non-interactive script

    return extracted_roi_path