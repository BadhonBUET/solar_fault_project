from ultralytics import YOLO
import cv2
import numpy as np

# ১. সঠিক ফোল্ডার নামসহ মডেল লোড করুন
model = YOLO(r'runs/detect/solar_train-3/weights/best.pt')

# ২. আপনার টেস্ট ছবির পাথ এখানে দিন
img_path = r"D:\OneDrive - BUET\Desktop\312\project\pv_thermal\ImageSet\train\images\1270_jpg.rf.a96091c2d82ad50ef8e811a77dfe5211.jpg"

img = cv2.imread(img_path)
if img is None:
    print(f"Image not found at: {img_path}")
    exit()

output_img = img.copy()

# YOLOv8 দিয়ে নির্দিষ্ট ছবিটিতে প্যানেলগুলো ডিটেক্ট করুন
results = model(img_path)

total_fault_count = 0

# ৩. প্যানেল বক্সের ভেতর ওপেনসিভি ফল্ট ডিটেকশন লজিক চালানো
for r in results:
    boxes = r.boxes
    for box in boxes:
        # প্যানেল বক্সের কোঅর্ডিনেট
        b = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, b)
        
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # [সংশোধন]: বর্ডারের রিফ্লেকশন এড়াতে প্যানেলের ভেতরের দিকে একটু মার্জিন (Margin) ছেড়ে ক্রপ করুন
        h_box = y2 - y1
        w_box = x2 - x1
        margin_y = int(h_box * 0.05) # ৫% মার্জিন
        margin_x = int(w_box * 0.05)
        
        crop_x1 = x1 + margin_x
        crop_y1 = y1 + margin_y
        crop_x2 = x2 - margin_x
        crop_y2 = y2 - margin_y
        
        panel_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if panel_crop.size == 0:
            continue
            
        gray_panel = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)
        blurred_panel = cv2.GaussianBlur(gray_panel, (5, 5), 0)
        
        # থ্রেশহোল্ড আপনার প্রয়োজন অনুযায়ী ১৪০ বা ১৫০ রাখতে পারেন
        _, thresh_fault = cv2.threshold(blurred_panel, 140, 255, cv2.THRESH_BINARY)
        fault_contours, _ = cv2.findContours(thresh_fault, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in fault_contours:
            area = cv2.contourArea(cnt)
            if 3 < area < 300:
                fx, fy, fw, fh = cv2.boundingRect(cnt)
                
                # গ্লোবাল পজিশন হিসাব করার সময় মার্জিন যুক্ত করতে হবে
                global_fx = crop_x1 + fx
                global_fy = crop_y1 + fy
                
                cv2.rectangle(output_img, (global_fx, global_fy), (global_fx + fw, global_fy + fh), (0, 0, 255), 2)
                total_fault_count += 1

# ৪. ফাইনাল রেজাল্ট সেভ করা
output_path = "hybrid_detected_result.jpg"
cv2.imwrite(output_path, output_img)
print(f"Success! Total faults detected inside panels: {total_fault_count}. Saved as '{output_path}'.")