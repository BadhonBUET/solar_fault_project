import cv2
import torch
import numpy as np
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Device and Model Setup
device = torch.device("cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2) # HotSpot and Normal
model = model.to(device)
model.eval()

# ResNet18 এর শেষ কনভোলিউশনাল লেয়ারটি সিলেক্ট করছি হিটম্যাপের জন্য
target_layers = [model.layer4[-1]]

# 2. Image Preprocessing for Grad-CAM
img_path = "D:\\OneDrive - BUET\\Desktop\\312\\dataset\\solar_fault_project\\dataset\\val\\HotSpot\\0raw_data_total_dataset_DJI_20230809120100_0005_T_jpg.rf.343c9121a3957656ded60b7b4a98316d.jpg" # এখানে আপনার টেস্ট ছবির পাথ দিন
rgb_img = cv2.imread(img_path)
if rgb_img is None:
    print("Image not found! Please check the path.")
    exit()

rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img_float = np.float32(rgb_img) / 255.0

# Tensor রূপান্তর
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = preprocess(rgb_img).unsqueeze(0).to(device)

# 3. Generating Grad-CAM Heatmap
cam = GradCAM(model=model, target_layers=target_layers)

# টার্গেট ক্লাস (HotSpot এর জন্য সাধারণত index 0 বা 1 হয়)
targets = None # None দিলে মডেল যেটিতে হাইয়েস্ট কনফিডেন্স পাবে সেটি ধরবে

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]

# মূল ছবির ওপর হিটম্যাপ ওভারলে করা (লাল/হলুদ রঙ দিয়ে হটস্পট মার্ক দেখাবে)
visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

# 4. Result Save and Display
output_path = "detected_result.jpg"
cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print(f"Success! Marked image saved as '{output_path}'. Open it to see the heatmap.")