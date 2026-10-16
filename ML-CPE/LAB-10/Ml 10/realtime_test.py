import os
import cv2
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

    # เปิดกล้อง Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Real-time Detection. Press 'q' to exit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.25)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Real-Time Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()