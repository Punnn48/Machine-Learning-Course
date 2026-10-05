import json
import os
import numpy as np

from data_loader import load_data
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data (2) .csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TEST_SIZE = 0.2
VAL_SIZE = 0.1
EPOCHS = 30
BATCH_SIZE = 32


def main():
    print("--" * 30)
    print("Neural Network: Heart Disease Tabular Prediction")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y, classes = load_data(DATA_PATH)

    np.save(f"{OUTPUT_DIR}/labels.npy", y)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )

    predictions = predict_model(model, X_test)

    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")
    print("Training and Evaluation Completed Successfully!")


if __name__ == "__main__":
    main()