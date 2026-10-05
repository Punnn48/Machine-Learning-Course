

This laboratory project implements a Feed-Forward Neural Network (Multilayer Perceptron) using TensorFlow/Keras to perform binary classification on tabular heart disease data. 


1. Data Preprocessing & Cleaning (`data_loader.py`, `preprocessing.py`, `split_data.py`):** 
   - Handles missing values by imputing with median statistics and encodes target labels for binary classification.
   - Standardizes input features using `StandardScaler` to ensure stable and efficient gradient descent.
   - Splits the dataset into stratified training, validation, and test sets to prevent data leakage and bias.
2. Model Architecture & Training (`cnn_model.py`):** 
   - Constructed using fully connected Dense layers with Batch Normalization and Dropout layers to prevent overfitting.
   - Utilizes callbacks (`EarlyStopping` and `ReduceLROnPlateau`) to dynamically optimize training epochs and learning rates.
3. Evaluation & Visualization (`evaluate.py`, `test_cnn.py`):** 
   - Evaluates performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
   - Generates automated training history graphs (Loss/Accuracy curves) and sample prediction summary tables saved in the `outputs/` directory.
