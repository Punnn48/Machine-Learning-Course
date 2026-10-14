import os
import numpy as np
import pandas as pd


def load_data(data_path):
    if os.path.isdir(data_path):
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"ไม่พบไฟล์ .csv ในโฟลเดอร์: {os.path.abspath(data_path)}")
        file_path = os.path.join(data_path, csv_files[0])
    else:
        file_path = data_path

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลที่เส้นทาง: {os.path.abspath(file_path)}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.replace('?', np.nan, inplace=True)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.fillna(df.median(numeric_only=True), inplace=True)

    target_col = 'num' if 'num' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    if len(np.unique(y)) == 2:
        y = (y > 0).astype(int)

    classes = ["Normal", "Heart Disease"]
    return X, y, classes