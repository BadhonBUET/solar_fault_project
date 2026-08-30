import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def select_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    return filedialog.askopenfilename(
        title="Select Solar Thermal Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )

def detect_hotspots_direct_dsp(image_path, output_path="detected_result.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        sys.exit("Failed to load image!")

    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Ignore dark background & pure black padding borders
    valid_mask = (gray > 30).astype(np.uint8) * 255

    # Step 2: Morphological Top-Hat Filter
    # Isolates small bright spots that are significantly brighter than their local background (15x15)
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Step 3: Extract Hotspot Candidates
    # Threshold local intensity difference (must be at least 35-40 gray levels brighter than surrounding pixel average)
    _, bright_seeds = cv2.threshold(tophat, 38, 255, cv2.THRESH_BINARY)
    
    # Also include extreme thermal saturation spots (Whiteout blooms > 210)
    _, saturation_seeds = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    
    combined_candidates = cv2.bitwise_or(bright_seeds, saturation_seeds)
    combined_candidates = cv2.bitwise_and(combined_candidates, valid_mask)

    # Step 4: Morphological Cleanup (Disconnect small noise dots)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(combined_candidates, cv2.MORPH_OPEN, clean_kernel)

    # Step 5: Filter Candidates by Hotspot Shape & Geometry
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotspot_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter 1: Area bounds (Hotspots are small & concentrated)
        if not (4 <= area <= 600):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Filter 2: Shape Check (Gravel noise forms long lines/streaks, hotspots are compact)
        if not (0.25 <= aspect_ratio <= 4.0):
            continue

        # Filter 3: Circularity / Compactness check
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        # Discard irregular jagged gravel edges (low circularity)
        if circularity < 0.20:
            continue

        # Calculate Confidence based on contrast difference
        roi = gray[y:y+h, x:x+w]
        avg_spot_intensity = np.mean(roi)
        confidence = min(max(((avg_spot_intensity - 120) / (255 - 120)) * 25 + 75, 75.0), 99.9)

        # Draw Red Target Boxes directly on verified hotspots
        cv2.rectangle(output_img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 2)
        cv2.putText(output_img, f"{confidence:.1f}%", (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        hotspot_count += 1

    cv2.imwrite(output_path, output_img)
    print(f"Success! Detected {hotspot_count} verified thermal hotspots directly.")

# Execution
print("Please select an image file for testing...")
input_file = select_image()
if input_file:
    detect_hotspots_direct_dsp(input_file)