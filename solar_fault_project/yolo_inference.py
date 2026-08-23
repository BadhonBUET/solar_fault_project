from ultralytics import YOLO
import cv2
import os

# প্রি-ট্রেইন্ড মডেল অথবা আপনার নিজস্ব ট্রেইন করা মডেল লোড করুন
# (শুরুতে টেস্ট করার জন্য 'yolov8n.pt' বা সোলার ফল্টের জন্য ডেটাসেট ট্রেইন করা মডেলের পাথ দিতে পারেন)
model = YOLO('yolov8n.pt') 

# আপনার টেস্ট ছবির পাথ
img_path = "D:\\OneDrive - BUET\\Desktop\\312\\dataset\\solar_fault_project\\dataset\\val\\HotSpot\\0raw_data_total_dataset_DJI_20230809120103_0006_T_jpg.rf.6c3e8434c02c80bcf6263441d9664c96.jpg"
if not os.path.exists(img_path):
    print("Image not found!")
    exit()

# প্রেডিকশন রান করা (conf বাড়িয়ে বা কমিয়ে সেনসিটিভিটি ঠিক করা যায়)
results = model(img_path, conf=0.35)

# ফল্ট বক্সগুলো ছবিতে ড্র করা
for r in results:
    im_array = r.plot(boxes=True, conf=True) # boxes=True মানে বক্সসহ দেখাবে

# রেজাল্ট সেভ করা
output_path = "yolo_precise_result.jpg"
cv2.imwrite(output_path, im_array)
print(f"Success! Precise detection saved as '{output_path}'.")