import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
N_SAMPLES = 4


def test_cnn(n_samples=N_SAMPLES):
    model = keras.models.load_model(f"{OUTPUT_DIR}/cnn_model.keras")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    probabilities = model.predict(X_sample, verbose=0)
    if probabilities.shape[-1] == 1:
        probabilities = probabilities.ravel()
        predictions = (probabilities > 0.5).astype(int)
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    else:
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

    print("\n--- Prediction Sample Results ---")
    for i in range(n_samples):
        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        print(f"[{i + 1}] Pred: {pred:<13} True: {true:<13} conf {confidence[i] * 100:5.1f}% {'OK' if correct else 'WRONG'}")

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('off')
    
    num_features_to_show = min(6, X_sample.shape[1])
    table_data = pd.DataFrame(X_sample[:, :num_features_to_show], columns=[f"Feature {j+1}" for j in range(num_features_to_show)])
    table_data['Prediction'] = [classes[p] for p in predictions]
    table_data['True Label'] = [classes[t] for t in y_sample]
    table_data['Status'] = ["OK" if p == t else "WRONG" for p, t in zip(predictions, y_sample)]
    
    table = ax.table(cellText=table_data.values, colLabels=list(table_data.columns), loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#4c72b0')
            cell.set_text_props(color='white', fontweight='bold')

    plt.title(f"Prediction Sample Results: {correct_total}/{n_samples} correct", fontsize=11, fontweight='bold', pad=15)

    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_cnn()