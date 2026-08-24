from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 1. Pop-up window to select image file easily
def select_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select Solar Thermal Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )
    return file_path

# 2. Load the OBB model with your correct path
model = YOLO(r'D:\OneDrive - BUET\Desktop\312\dataset\solar_fault_project\dataset_obb\runs\obb\solar_obb_run-10\weights\best.pt')

# 3. Select image using dialog box
print("Please select your test image from the popped-up window...")
img_path = select_image()

if not img_path:
    print("No image selected! Terminating program.")
    exit()

print(f"Selected Image: {img_path}")
img = cv2.imread(img_path)

if img is None:
    print("Error: Failed to load the image!")
    exit()

output_img = img.copy()

# 4. Run OBB prediction
results = model(img, conf=0.45, iou=0.5)
total_fault_count = 0

for r in results:
    # Let Ultralytics draw the rotated OBB boxes for the panels
    output_img = r.plot() 

    if r.obb is not None:
        boxes = r.obb.xyxyxyxy  # 4 corner coordinates of rotated panels
        
        for box in boxes:
            pts = box.cpu().numpy().astype(np.int32)
            
            # --- ৫% ইনার মার্জিন বা ইনওয়ার্ড করার জন্য পলিগনের সেন্টার বের করে একটু ছোট করা ---
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # প্যানেলের চার কোণাকে সেন্টারের দিকে প্রায় ৫% সংকুচিত (shrink) করা
                shrink_pts = []
                for pt in pts:
                    # সেন্টারের দিকে পয়েন্টগুলো সামান্য টেনে আনা
                    sx = int(cX + (pt[0] - cX) * 0.93)  # 0.93 মানে প্রায় ৭% ভেতরের দিকে
                    sy = int(cY + (pt[1] - cY) * 0.93)
                    shrink_pts.append([sx, sy])
                pts_inner = np.array(shrink_pts, dtype=np.int32)
            else:
                pts_inner = pts

            # Create a precise mask for the inner rotated panel
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts_inner], 255)
            
            # Get bounding rect to crop the inner panel region
            x, y, w, h = cv2.boundingRect(pts_inner)
            x, y = max(0, x), max(0, y)
            panel_crop = img[y:y+h, x:x+w]
            mask_crop = mask[y:y+h, x:x+w]
            
            if panel_crop.size == 0:
                continue
                
            # Grayscale & Thresholding strictly inside the inner panel
            gray_panel = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)
            blurred_panel = cv2.GaussianBlur(gray_panel, (5, 5), 0)
            
            threshold_value = 190  
            _, thresh_fault = cv2.threshold(blurred_panel, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Apply mask to ensure detections stay strictly INSIDE the inner area
            thresh_fault = cv2.bitwise_and(thresh_fault, thresh_fault, mask=mask_crop)
            
            fault_contours, _ = cv2.findContours(thresh_fault, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in fault_contours:
                area = cv2.contourArea(cnt)
                if 5 < area < 300:
                    fx, fy, fw, fh = cv2.boundingRect(cnt)
                    aspect_ratio = float(fw) / fh if fh > 0 else 0
                    
                    if 0.3 < aspect_ratio < 3.0:
                        fault_roi = gray_panel[fy:fy+fh, fx:fx+fw]
                        if fault_roi.size > 0:
                            avg_intensity = np.mean(fault_roi)
                            confidence = min(max(((avg_intensity - threshold_value) / (255 - threshold_value)) * 20 + 80, 75), 99)
                        else:
                            confidence = 80.0

                        if confidence >= 82.0:  
                            global_fx = x + fx
                            global_fy = y + fy
                            
                            cv2.rectangle(output_img, (global_fx, global_fy), (global_fx + fw, global_fy + fh), (0, 0, 255), 1)
                            cv2.putText(output_img, f"{confidence:.1f}%", (global_fx, max(global_fy - 3, 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                            total_fault_count += 1

# 5. Save and finish
output_path = "detected_result.jpg"
cv2.imwrite(output_path, output_img)
print(f"Processing complete! Saved as '{output_path}'. Verified faults: {total_fault_count}")