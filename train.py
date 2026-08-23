from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# সরাসরি ফুল পাথ বা সঠিক ফাইলের নাম দিন
results = model.train(data=r'D:\OneDrive - BUET\Desktop\312\dataset\solar_fault_project\data.yaml', epochs=50, imgsz=640, name='solar_train')