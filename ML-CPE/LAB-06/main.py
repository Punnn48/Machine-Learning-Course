import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data
from preprocessing import preprocess_data
from split_data import split_dataset
from nn_model import build_nn_model
from sklearn.metrics import accuracy_score

# สร้างโฟลเดอร์ outputs สำหรับเก็บรูปกราฟ
os.makedirs("outputs", exist_ok=True)

# 1. โหลดข้อมูล CSV
df = load_data("data (2) (1).csv")
target_col = 'num'

# 2. Preprocessing & Standardization
X, y, scaler = preprocess_data(df, target_col)

# 3. Split Data (Train, Test)
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

# รวม Train และ Val เข้าด้วยกัน
X_train_full = np.concatenate((X_train, X_val), axis=0)
y_train_full = np.concatenate((y_train, y_val), axis=0)

# 4. ทดลองเปรียบเทียบ Configurations และจำนวน Epochs (max_iter)
configs = {
    "Config 1 (16,)": (16,),
    "Config 2 (32, 16)": (32, 16),
    "Config 3 (64, 32, 16)": (64, 32, 16)
}

epochs_list = [20, 50, 100]
results = []

print("\n--- เริ่มต้นการเทรนและประเมินผล Neural Network (MLP) ---")
for config_name, hidden_layers in configs.items():
    print(f"\n[Testing] {config_name}")
    for epochs in epochs_list:
        model = build_nn_model(hidden_layers=hidden_layers, max_iter=epochs)
        
        # Train model
        model.fit(X_train_full, y_train_full)
        
        # Predict และประเมินผลบน Test Set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Config': config_name,
            'Epochs': epochs,
            'Accuracy': acc
        })
        print(f" -> Epochs/Iterations: {epochs:3d} | Test Accuracy: {acc:.4f}")

# 5. สร้างกราฟแสดงผลเปรียบเทียบ
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
for config_name in configs.keys():
    subset = results_df[results_df['Config'] == config_name]
    plt.plot(subset['Epochs'], subset['Accuracy'], marker='o', linewidth=2, label=config_name)

plt.title('Comparison of Neural Network Configurations and Epochs', fontsize=14)
plt.xlabel('Epochs (Max Iterations)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# บันทึกรูปภาพลงในโฟลเดอร์ outputs
graph_path = "outputs/accuracy_comparison.png"
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
print(f"\nบันทึกกราฟเรียบร้อยแล้วที่: {graph_path}")

# แสดงกราฟขึ้นมาดู (หากรันในEnvironmentที่มีหน้าจอ GUI)
# plt.show()

print("\nการรันโปรเจกต์เสร็จสิ้นสมบูรณ์!")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# สร้างโมเดลตัวอย่างที่ดีที่สุด (เช่น Config ที่ดีที่สุด) มา plot Confusion Matrix
best_model = build_nn_model(hidden_layers=(32, 16), max_iter=100)
best_model.fit(X_train_full, y_train_full)
y_pred_best = best_model.predict(X_test)

# คำนวณ Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)

# วาดรูป Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.title('Confusion Matrix', fontsize=14)

# บันทึกรูปภาพ
cm_path = "outputs/confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"บันทึก Confusion Matrix เรียบร้อยแล้วที่: {cm_path}")
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# 6. ฟังก์ชันสำหรับสุ่มแสดงตัวอย่างการทำนาย
def plot_prediction_samples(model, X_test, y_test, num_samples=4):
    """สุ่มตัวอย่างข้อมูลทดสอบ num_samples รูป และแสดงผลการทำนาย"""
    # สุ่ม index
    num_test_samples = len(y_test)
    if num_samples > num_test_samples:
        num_samples = num_test_samples
        
    indices = random.sample(range(num_test_samples), num_samples)
    
    # เตรียม subplot
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Prediction Samples from Test Set', fontsize=16)
    
    # เนื่องจากข้อมูลของเราเป็นตาราง ไม่ใช่รูปภาพ เราจะแสดงค่า Feature บางส่วน
    # หรือหากต้องการให้คล้ายตัวอย่างอาจารย์ที่สุด เราจะแปลงข้อมูลกลับเป็น DataFrame ชั่วคราว
    # เพื่อดึงข้อมูลดิบมาแสดง (ในกรณีที่ X_test ถูก scale แล้ว)
    
    # ตัวอย่างนี้จะแสดงค่า feature แรกๆ และ label
    for i, index in enumerate(indices):
        ax = fig.add_subplot(1, num_samples, i + 1)
        
        # ดึงข้อมูลตัวอย่าง (แบบ Unscaled หากเป็นไปได้)
        # ในที่นี้จะแสดงแค่บาง Feature (เช่น age, trestbps, chol) เป็นตัวอย่าง
        # หรือถ้าต้องการง่ายๆ แสดง vector เลยก็ได้
        
        # สมมติเรามี scaler อยู่แล้ว สามารถ inverse_transform ได้
        # แต่ต้องระวังเรื่อง dimension ของ X
        # X_original_sample = scaler.inverse_transform(X_test[index].reshape(1, -1))
        
        # แสดงผลแบบง่ายๆ คือการพิมพ์ค่า feature บางส่วน
        sample_data_subset = X_test[index][:3] # แสดง 3 feature แรก
        true_label = y_test[index]
        pred_label = model.predict(X_test[index].reshape(1, -1))[0]
        
        # ใส่สีเขียวถ้าถูก, แดงถ้าผิด
        color = 'green' if pred_label == true_label else 'red'
        
        ax.text(0.5, 0.7, f'Pred: {pred_label}\nTrue: {true_label}', 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=12, color=color, transform=ax.transAxes)
        
        # ซ่อนแกน
        ax.axis('off')
        ax.set_title(f'Sample Index: {index}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # บันทึกรูป
    sample_path = "outputs/prediction_samples.png"
    plt.savefig(sample_path)
    print(f"บันทึกรูปตัวอย่างการทำนายเรียบร้อยแล้วที่: {sample_path}")
    # plt.show() # แสดงกราฟหากรันใน Environment ที่เหมาะสม

# 7. สร้างโมเดลตัวอย่างที่ดีที่สุด และรันฟังก์ชัน
best_hidden_layers = (32, 16)
best_epochs = 100
final_model = build_nn_model(hidden_layers=best_hidden_layers, max_iter=best_epochs)
final_model.fit(X_train_full, y_train_full)

plot_prediction_samples(final_model, X_test, y_test)