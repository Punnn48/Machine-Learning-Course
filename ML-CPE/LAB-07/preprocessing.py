import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_tabular_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def to_features(data):
    return np.ascontiguousarray(data, dtype=np.float32)