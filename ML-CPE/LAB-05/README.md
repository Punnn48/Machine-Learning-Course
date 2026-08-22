Importing Libraries

First, I imported all the necessary libraries like pandas for reading our CSV file, numpy for handling numerical data, matplotlib for plotting graphs, and scikit-learn modules for scaling, training, and evaluating the Support Vector Machine (SVM) model.

Loading and Cleaning the Dataset

I loaded the heart disease dataset (data (2) (1).csv) using pandas and cleaned up the column names by removing extra spaces.

Since the dataset contains missing values marked with ?, I replaced them with NaN and dropped columns that had too many missing values (like slope, ca, and thal).

Then, I converted all remaining data into numeric format and filled any missing values using the median value of each column.

Data Preprocessing & Target Formulation

I separated the features (X) from the target variable (y).

I converted the num target column into a binary classification problem (0 for healthy, and 1 for having heart disease) so the SVM model could process it easily.

Data Splitting & Feature Scaling

I split the dataset into training and testing sets using an 80/20 ratio with train_test_split.

After that, I applied StandardScaler to normalize all the features so that they are on the same scale, which is super important for SVM to find the best decision boundary.

Model Training & Kernel Evaluation

I set up a loop to train and evaluate SVM models using three different kernels: Linear, Polynomial (poly), and RBF (SVC).

For each kernel, I trained the model on the scaled training data, made predictions on the test set, and calculated the accuracy score.

Finally, I printed out the accuracy for each kernel and checked which one gave the highest performance.

Visualization

Lastly, I plotted a bar chart using matplotlib to visually compare the accuracy scores across the three different SVM kernels to see how well each one performed.
