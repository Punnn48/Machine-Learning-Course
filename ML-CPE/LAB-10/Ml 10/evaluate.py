import os
from ultralytics import YOLO

def main():
    # ตรวจสอบว่ามีไฟล์ที่เทรนแล้วหรือไม่ ถ้าไม่มีให้ใช้ yolov8n.pt ชั่วคราว
    model_path = "runs/detect/train_exp/weights/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Loading trained model from: {model_path}")
    else:
        print(f"Warning: Not found '{model_path}'. Using pretrained 'yolov8n.pt' instead.")
        model = YOLO("yolov8n.pt")

    # ประเมินผล
    metrics = model.val(data="dataset/data.yaml", split="val")

    print("\n--- Model Evaluation Results ---")
    print(f"Precision : {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall    : {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP50     : {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP50-95  : {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

if __name__ == "__main__":
    main()