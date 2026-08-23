import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==========================================
# ১. Settings & Model Setup
# ==========================================
device = torch.device("cpu")
class_names = ['HotSpot', 'Normal'] # আপনার ফোল্ডারের নাম অনুযায়ী

# Load architecture
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# (ঐচ্ছিক) আপনি চাইলে মডেল সেভ করে সেটি লোড করতে পারেন, 
# তবে এই মুহূর্তে জাস্ট স্ট্রাকচার চেক করার জন্য কোডটি সাজানো হয়েছে।

model = model.to(device)
model.eval()

# ==========================================
# ২. Single Image Prediction Function
# ==========================================
def predict_image(image_path):
    # Image preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)
    input_tensor = input_tensor.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0]]

    print(f"\n--- Prediction Result ---")
    print(f"Image: {image_path}")
    print(f"Predicted Fault Category: **{predicted_class}**\n")

# ==========================================
# ৩. Run Prediction
# ==========================================
if __name__ == '__main__':
    # আপনার যেকোনো একটি ছবির পাথ এখানে দিন
    test_img_path = r"D:\OneDrive - BUET\Desktop\312\project\pv_thermal\ImageSet\train\images\0raw_data_total_dataset_DJI_20230809120018_0011_T_jpg.rf.5b78e022f4e8099f9778160f3ec73990.jpg"
    
    predict_image(test_img_path)