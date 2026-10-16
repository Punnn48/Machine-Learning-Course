import os
import glob
import random
from ultralytics import YOLO

def main():
    # ตรวจสอบไฟล์โมเดล
    model_path = "runs/detect/train_exp/weights/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Loading trained model from: {model_path}")
    else:
        print(f"Warning: Not found '{model_path}'. Using pretrained 'yolov8n.pt' instead.")
        model = YOLO("yolov8n.pt")

    # ค้นหารูปในโฟลเดอร์ test
    test_images = glob.glob("dataset/test/images/*.jpg") + glob.glob("dataset/test/images/*.png")
    if not test_images:
        print("Error: No test images found in dataset/test/images/")
        return

    sample_img = random.choice(test_images)

    # ทำนายผล
    results = model.predict(source=sample_img, conf=0.25, save=True, project="runs/detect", name="image_test_results")
    print(f"Detection completed for {sample_img}. Results saved in runs/detect/image_test_results")

if __name__ == "__main__":
    main()