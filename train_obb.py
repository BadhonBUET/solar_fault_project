from ultralytics import YOLO

def main():
    # 1. Load the pre-trained YOLOv8 OBB model
    model = YOLO("yolov8n-obb.pt")

    # 2. Train the model using your OBB dataset yaml file
    results = model.train(
        data="dataset_obb.yaml",  
        epochs=50,                
        imgsz=640,                
        batch=4,                  # যেহেতু CPU তে ট্রেন করবেন, ব্যাচ সাইজ ছোট (যেমন ৪ বা ৮) রাখা ভালো
        device='cpu',             # এখানে '0'-এর বদলে 'cpu' করে দেওয়া হয়েছে
        workers=2,                # CPU ট্রেইনিংয়ের জন্য ওয়ার্কার্স সংখ্যা ২ বা ৪ রাখতে পারেন
        save=True,                
        project="runs/obb",       
        name="solar_obb_run"      
    )

    print("Training finished successfully!")

if __name__ == '__main__':
    main()