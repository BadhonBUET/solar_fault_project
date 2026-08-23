import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==========================================
# ১. DSP Preprocessing Function
# ==========================================
def dsp_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    filtered = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    rgb_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb_img

# ==========================================
# ২. Dataset & Data Loaders
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Detected Fault Categories: {class_names}")
print(f"Dataset Sizes: {dataset_sizes}")

# Device configuration
device = torch.device("cpu") # আপাতত সিপিইউ দিয়েই চালাচ্ছি, চাইলে cuda দিতে পারেন
print(f"Using Device: {device}")

# ==========================================
# ৩. Model Setup (Transfer Learning with ResNet18)
# ==========================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Modify final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ==========================================
# ৪. Real Training Loop
# ==========================================
if __name__ == '__main__':
    print("\nStarting Real Model Training...")
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 25)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass & optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print(f"Train Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    print("\nTraining completed successfully! Model is learning from your images.")