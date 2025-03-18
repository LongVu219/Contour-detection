from ultralytics import YOLO
import os
import yaml

data_yaml_path = '/home/longvv/Contour-detection/yolo_dataset/data.yaml'
model_type = 'yolo11n' 
epochs = 30
batch_size = 32
img_size = 640


os.makedirs('runs/train_ultralytics', exist_ok=True)
model = YOLO(model_type)  


results = model.train(
    data=data_yaml_path,
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    project='runs/train_ultralytics',
    name='yolo11n',
)

test_results = model.val(
    data=data_yaml_path,
    save=True,
    save_txt=True,
    save_conf=True,
    project='runs/test',
    name='yolo11n_contour_test'
)

print(f"Test completed and results saved to 'runs/test/yolo11n_contour_test'")