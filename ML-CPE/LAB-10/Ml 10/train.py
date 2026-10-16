from ultralytics import YOLO

def main():
    # โหลด Pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # ฝึกสอนโมเดล (ปรับ epochs, imgsz, batch ตามต้องการ)
    results = model.train(
        data="dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="train_exp",
        save=True
    )
    print("Training Completed Successfully!")

if __name__ == "__main__":
    main()