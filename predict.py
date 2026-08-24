from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 1. Function to open file dialog and select an image
def select_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select Solar Thermal Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )
    return file_path

# 2. Load the YOLO model with the correct folder path
model = YOLO(r'runs/detect/solar_train-3/weights/best.pt')

# 3. Prompt user to select an image file
print("Please select an image file for testing...")
img_path = select_image()

if not img_path:
    print("No image selected! Terminating program.")
    exit()

print(f"Selected Image: {img_path}")

img = cv2.imread(img_path)
if img is None:
    print("Failed to load the image!")
    exit()

output_img = img.copy()

# 4. Detect panels in the selected image using YOLOv8
results = model(img_path)

total_fault_count = 0

# 5. Run OpenCV fault detection logic inside the detected panel boxes
for r in results:
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, b)
        
        # Get panel confidence score
        conf = box.conf[0].item()
        panel_label = f"Panel {conf:.2f}"
        
        # Draw green border and label for the panel
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_img, panel_label, (x1, max(y1 - 5, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Calculate margins to avoid border reflections
        h_box = y2 - y1
        w_box = x2 - x1
        margin_y = int(h_box * 0.05)
        margin_x = int(w_box * 0.05)
        
        crop_x1 = x1 + margin_x
        crop_y1 = y1 + margin_y
        crop_x2 = x2 - margin_x
        crop_y2 = y2 - margin_y
        
        panel_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if panel_crop.size == 0:
            continue
            
        # Standard Grayscale conversion (Without CLAHE)
        gray_panel = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)
        blurred_panel = cv2.GaussianBlur(gray_panel, (5, 5), 0)
        
        # Thresholding (You can adjust threshold value here, e.g., 150 or 180)
        threshold_value = 170
        _, thresh_fault = cv2.threshold(blurred_panel, threshold_value, 255, cv2.THRESH_BINARY)
        fault_contours, _ = cv2.findContours(thresh_fault, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in fault_contours:
            area = cv2.contourArea(cnt)
            
            # Area filter for compact hotspots
            if 3 < area < 400:
                fx, fy, fw, fh = cv2.boundingRect(cnt)
                
                # Aspect ratio check
                aspect_ratio = float(fw) / fh if fh > 0 else 0
                
                # Exclude panel grid/busbar lines
                panel_h, panel_w = gray_panel.shape
                is_grid_line = (fw > panel_w * 0.15) or (fh > panel_h * 0.15)
                
                if not is_grid_line and (0.2 < aspect_ratio < 5.0):
                    # Calculate confidence percentage based on intensity
                    fault_roi = gray_panel[fy:fy+fh, fx:fx+fw]
                    if fault_roi.size > 0:
                        avg_intensity = np.mean(fault_roi)
                        confidence = min(max(((avg_intensity - threshold_value) / (255 - threshold_value)) * 20 + 80, 75), 99)
                    else:
                        confidence = 80.0

                    # Keep faults with >= 80% confidence
                    if confidence >= 80.0:
                        global_fx = crop_x1 + fx
                        global_fy = crop_y1 + fy
                        
                        # Draw thin red box and percentage score
                        cv2.rectangle(output_img, (global_fx, global_fy), (global_fx + fw, global_fy + fh), (0, 0, 255), 1)
                        
                        fault_label = f"{confidence:.1f}%"
                        cv2.putText(output_img, fault_label, (global_fx, max(global_fy - 3, 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                        total_fault_count += 1

# 6. Save the final result
output_path = "detected_result.jpg"
cv2.imwrite(output_path, output_img)
print(f"Success! Total verified faults (>=80%) inside panels: {total_fault_count}. Saved as '{output_path}'.")