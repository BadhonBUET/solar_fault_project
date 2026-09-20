import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 1. Image Selection Utility
# ----------------------------------------------------------------------
def select_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select Solar Thermal Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    root.destroy()
    return file_path

# ----------------------------------------------------------------------
# 2. Local DSP Functions (Variance Profile)
# ----------------------------------------------------------------------
def local_std_map(gray, ksize=13):
    gray_f = gray.astype(np.float32)
    k = (ksize, ksize)
    mean = cv2.blur(gray_f, k)
    mean_sq = cv2.blur(gray_f ** 2, k)
    var = np.maximum(mean_sq - mean ** 2, 0)
    std = np.sqrt(var)
    return cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def compute_dsp_panel_mask(img_gray):
    h, w = img_gray.shape
    std_norm = local_std_map(img_gray, ksize=13)
    row_profile = np.mean(std_norm, axis=1)
    threshold_val = np.percentile(row_profile, 40)
    
    panel_rows = (row_profile <= threshold_val).astype(np.uint8)
    panel_mask_2d = np.repeat(panel_rows[:, np.newaxis], w, axis=1) * 255
    panel_mask_2d = cv2.morphologyEx(
        panel_mask_2d, cv2.MORPH_CLOSE, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    )
    return std_norm, panel_mask_2d

# ----------------------------------------------------------------------
# 3. Execution Stream
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Model Loading
    model_path = r'D:\OneDrive - BUET\Desktop\312\dataset\solar_fault_project\dataset_obb\runs\obb\solar_obb_run-10\weights\best.pt'
    
    if not os.path.exists(model_path):
        print(f"Warning: Model path not found: {model_path}")
        print("Falling back to standard loading or user path.")
    
    model = YOLO(model_path)

    print("Please select your thermal image...")
    img_path = select_image()

    if not img_path:
        print("No image selected! Exiting.")
        sys.exit()

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Error: Could not load image file.")
        sys.exit()

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # --- Step A: Compute DSP Panel Mask & Standard Deviation Map ---
    std_map, dsp_panel_mask = compute_dsp_panel_mask(img_gray)

    # --- Step B: Run YOLO OBB Inference for ML Panel Mask ---
    results = model(img_bgr, conf=0.45, iou=0.5)
    
    ml_panel_mask = np.zeros((h, w), dtype=np.uint8)
    annotated_img = img_bgr.copy()

    # Iterate over predictions and construct the ML polygon panel mask correctly
    for r in results:
        if r.obb is not None and len(r.obb) > 0:
            # Extract corner points: shape (N, 4, 2)
            obb_corners = r.obb.xyxyxyxy.cpu().numpy().astype(np.int32)

            for pts in obb_corners:
                # Fill ML panel mask with exact rotated bounding box region
                cv2.fillPoly(ml_panel_mask, [pts], 255)
                
                # Draw outer rotated OBB contour on visualization image (Green)
                cv2.polylines(annotated_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # --- Step C: Calculate Overlapped Region (Final Panel Area) ---
    final_panel_area = cv2.bitwise_and(dsp_panel_mask, ml_panel_mask)

    # --- Step D: Hotspot Extraction strictly within Final Panel Area ---
    total_fault_count = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    
    threshold_value = 35
    _, seeds = cv2.threshold(tophat, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Restrict hotspot extraction ONLY within the overlapped system panel area
    valid_seeds = cv2.bitwise_and(seeds, final_panel_area)
    fault_contours, _ = cv2.findContours(valid_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in fault_contours:
        area = cv2.contourArea(cnt)
        if 5 <= area <= 400:
            fx, fy, fw, fh = cv2.boundingRect(cnt)
            aspect_ratio = float(fw) / fh if fh > 0 else 0
            
            if 0.3 < aspect_ratio < 3.0:
                fault_roi = img_gray[fy:fy+fh, fx:fx+fw]
                if fault_roi.size > 0:
                    avg_intensity = np.mean(fault_roi)
                    confidence = min(max(((avg_intensity - threshold_value) / (255 - threshold_value)) * 20 + 80, 75), 99.9)
                else:
                    confidence = 80.0

                if confidence >= 82.0:
                    # Draw red bounding rectangle for verified hotspot
                    cv2.rectangle(annotated_img, (fx - 2, fy - 2), (fx + fw + 2, fy + fh + 2), (0, 0, 255), 2)
                    cv2.putText(annotated_img, f"{confidence:.1f}%", (fx - 2, max(fy - 4, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    total_fault_count += 1

    print(f"Processing complete. Hotspots identified: {total_fault_count}")

    # Save Annotated Image Output
    cv2.imwrite("detected_result.jpg", annotated_img)

    # --- Step E: 2x3 Plotting Subplot Grid ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Main Image
    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Main Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Variance Image
    axes[0, 1].imshow(std_map, cmap='magma')
    axes[0, 1].set_title('2. Variance Image', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Panel Mask
    axes[0, 2].imshow(dsp_panel_mask, cmap='gray')
    axes[0, 2].set_title('3. Panel Mask (DSP)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # 4. ML Detected Panel
    axes[1, 0].imshow(ml_panel_mask, cmap='gray')
    axes[1, 0].set_title('4. ML Detected Panel (YOLO OBB)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # 5. Final Panel Area
    axes[1, 1].imshow(final_panel_area, cmap='gray')
    axes[1, 1].set_title('5. Final Panel Area (Overlapped)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # 6. Hotspots Inside Panel
    axes[1, 2].imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('6. Hotspots Inside Panel', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()