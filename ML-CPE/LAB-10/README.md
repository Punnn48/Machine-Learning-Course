Dataset Preparation & Configuration (data.yaml):

Prepares and structures the dataset in YOLO annotation format with normalized bounding box coordinates (<class_id> <x_center> <y_center> <width> <height>).

Configures dataset paths and defines object classes (person, rider, car, bus, truck, bike, motor, traffic light, traffic sign, train).

Model Training & Optimization (train.py):

Fine-tunes a pretrained yolov8n.pt model across specified training epochs, image resolution, and batch sizes.

Utilizes internal loss functions and dynamic learning rate schedulers, saving the best checkpoint weights to runs/detect/train_exp/weights/best.pt.

Evaluation & Performance Metrics (evaluate.py):

Evaluates validation and test splits using standard metrics including Precision, Recall, mAP50, and mAP50-95.

Generates quantitative performance logs and summary statistics for each training configuration.

Image & Real-Time Inference (image_test.py, realtime_test.py):

Loads trained checkpoint weights to perform inference on static test images, drawing annotated bounding boxes with confidence scores.

Processes live webcam video streams for real-time frame-by-frame object detection and visual display.
