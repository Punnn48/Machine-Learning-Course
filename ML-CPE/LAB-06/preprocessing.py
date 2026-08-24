import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_column):
    """ทำความสะอาดข้อมูล แทนที่ '?' ด้วย NaN และเติมค่าด้วย Median"""
    # 1. ทำความสะอาดชื่อคอลัมน์ตัดช่องว่างส่วนเกิน
    df.columns = df.columns.str.strip()
    
    # 2. แทนที่เครื่องหมาย '?' ด้วย NaN
    df = df.replace('?', np.nan)
    
    # 3. แปลงทุกคอลัมน์ให้เป็นตัวเลข โดยใช้ errors='coerce' ป้องกัน Error จากตัวอักษร/เครื่องหมาย
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 4. เติมค่าที่หายไปด้วยค่ามัธยฐาน (Median) ของแต่ละคอลัมน์
    df = df.fillna(df.median())

    # 5. แยก Features (X) และ Target (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # 6. Standardize ข้อมูลตัวเลขด้วย StandardScaler ตามโจทย์แล็บ
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler