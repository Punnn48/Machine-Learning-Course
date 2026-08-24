import pandas as pd
import os

def load_data(file_path):
    """โหลดข้อมูลจากไฟล์ CSV"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ที่กำหนด: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"โหลดข้อมูลสำเร็จ! ขนาดข้อมูล: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data("data (2) (1).csv")
    print(df.head())