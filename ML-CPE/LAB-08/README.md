Data Loading & Cleaning (data_loader.py)

Load the tabular heart disease dataset from data (2) .csv.

Clean column headers and replace any missing or invalid values (such as '?') with NaN.

Impute missing numeric values using column median statistics and encode target labels for binary classification.

Feature Scaling (preprocessing.py)

Apply StandardScaler to normalize all continuous feature columns, ensuring zero mean and unit variance for stable and efficient neural network convergence.

Dataset Splitting (split_data.py)

Split the processed dataset into Training (70%), Validation (10%), and Testing (20%) sets using stratified sampling to maintain class balance.

Model Architecture & Training (vgg_model.py)

Construct a Deep Neural Network (Multilayer Perceptron) using Keras with Dense layers, Batch Normalization, and Dropout layers to prevent overfitting.

Compile the model with the Adam optimizer and binary cross-entropy loss function.

Implement training callbacks (EarlyStopping and ReduceLROnPlateau) to optimize training efficiency and dynamically adjust the learning rate.

Save the trained model weights as vgg_model.keras inside the outputs/ folder.

Model Evaluation & Visualization (evaluate.py)

Evaluate performance on the test set using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix metrics.

Generate and save training history curves (Loss/Accuracy) as well as the confusion matrix plot.

Sample Testing & Visualization (test_vgg.py)

Load the saved model and test set to run predictions on randomly selected samples.

Generate a visual summary table comparing model predictions against true labels and save it as prediction_sample.
