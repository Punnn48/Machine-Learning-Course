Data Loading (data_loader.py)
Description: Loads the raw dataset from the specified CSV file (data (2) (1).csv) into a Pandas DataFrame.

Key Action: Inspects the structure of the dataset and reads all features and target columns for further processing.

2. Data Preprocessing (preprocessing.py)
Description: Cleans and prepares the raw data to ensure high model quality.

Key Actions:

Strips whitespace from column names.

Replaces missing or invalid character placeholders (such as '?') with NaN.

Converts all feature columns into numeric data types and handles missing values by imputing them with the median value of each column.

Normalizes the feature values using StandardScaler to ensure a mean of 0 and a standard deviation of 1, which helps the Neural Network converge faster and more accurately.

3. Data Splitting (split_data.py)
Description: Divides the processed dataset into training, validation, and testing subsets to train the model and evaluate its generalization performance.

4. Neural Network Modeling & Training (nn_model.py & main.py)
Description: Constructs a Multi-Layer Perceptron (MLP) Classifier using different configurations and epoch sizes.

Key Actions:

Evaluates multiple hidden layer architectures (e.g., Config 1: (16,), Config 2: (32, 16), Config 3: (64, 32, 16)).

Trains the model using varying maximum iterations (epochs: 20, 50, 100) with the Adam optimizer.

Measures the Test Accuracy for each configuration to determine the optimal model structure.

5. Evaluation & Visualization (outputs/)
Description: Generates performance metrics and visual plots to analyze the results.

Key Visualizations Generated:

Accuracy Comparison Chart (accuracy_comparison.png): Illustrates how different hidden layer configurations and epoch sizes affect model accuracy.

Confusion Matrix (confusion_matrix.png): Evaluates classification performance by displaying true positives, false positives, true negatives, and false negatives.

Prediction Samples (prediction_samples.png): Displays random test samples comparing predicted labels against actual true labels.
