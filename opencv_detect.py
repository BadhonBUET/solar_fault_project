import cv2
import numpy as np
import os

# 1. Load Image
img_path = r"D:\OneDrive - BUET\Desktop\312\project\pv_thermal\ImageSet\train\images\0raw_data_total_dataset_DJI_20230809115745_0013_T_jpg.rf.4a50fdbf5a540d8a3c26d6ed2d3fd1d2.jpg" # আপনার ছবির পাথ দিন
img = cv2.imread(img_path)
if img is None:
    print("Image not found!")
    exit()

output_img = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Automatic Grid/Panel Masking via Edge Detection & Contours
# সোলার প্যানেলের ধার বা বর্ডারগুলো খুঁজে বের করার জন্য Canny Edge ব্যবহার করছি
edges = cv2.Canny(gray, 50, 150)

# প্যানেলের ভেতরের বড় রেক্টেঙ্গুলার ব্লকগুলো বাউন্ডারি দিয়ে আলাদা করা
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(gray)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # প্যানেলের ব্লকের আকৃতি অনুযায়ী ফিল্টার (ব্যাকগ্রাউন্ডের ঘাস বা ছোট নয়েজ বাদ দেওয়ার জন্য)
    if w > 80 and h > 20: 
        # প্যানেল এলাকার ভেতরটা সাদা করে একটি মাস্ক তৈরি করছি
        cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

# যদি কোনো কারণে মাস্ক ছোট হয় বা না পায়, তবে পুরো ছবি ধরে কাজ করবে যেন ক্রাশ না করে
if cv2.countNonZero(mask) == 0:
    mask = np.ones_like(gray) * 255

# 3. Detect Faults (Hotspots) ONLY inside the Automatically Detected Panel Mask
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh_fault = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

# প্যানেল মাস্কের সাথে ফল্ট থ্রেশহোল্ড এন্ড (AND) অপারেশন করা, 
# যাতে প্যানেলের বাইরের কোনো কিছু কখনোই কাউন্ট না হয়।
masked_fault = cv2.bitwise_and(thresh_fault, thresh_fault, mask=mask)

fault_contours, _ = cv2.findContours(masked_fault, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
fault_count = 0

for cnt in fault_contours:
    area = cv2.contourArea(cnt)
    if 3 < area < 300: # ফল্টের আকৃতি ফিল্টার
        fx, fy, fw, fh = cv2.boundingRect(cnt)
        cv2.rectangle(output_img, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
        fault_count += 1

# 4. Save Result
output_path = "opencv_detected_result.jpg"
cv2.imwrite(output_path, output_img)
print(f"Success! Automatically detected faults inside panels: {fault_count}. Saved as '{output_path}'.")