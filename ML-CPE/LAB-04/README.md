mporting Libraries
First, I imported all the necessary libraries like pandas for handling our CSV file, numpy for math stuff, matplotlib for plotting graphs, and tools from scikit-learn for building the KNN model.
Loading the Dataset
I loaded our hotel dataset (Travel_Kayak (2).csv) using pandas so I could check out what the data looks like.
Data Preprocessing & Target Creation
Since our target column (historic_price) was continuous numbers, I used pd.qcut to split the prices into 3 neat categories (Low, Medium, High) so we could actually do a classification task.
Then, I dropped columns that we don't need (like hotel names, IDs, and text columns) and converted the remaining categorical features into numbers using One-Hot Encoding (pd.get_dummies).
Splitting and Scaling Data
I split the dataset into training and testing sets using an 80/20 ratio with train_test_split.
After that, I used StandardScaler to normalize all the feature values so that the distance calculation in KNN won't get messed up by different scales.
Training and Evaluating the KNN Model
I set up a loop to test different numbers of neighbors (k=3,5,7) using KNeighborsClassifier.
For each k, I trained the model on the training data, made predictions on the test set, and calculated the accuracy score.
Finally, I printed out the accuracy for each k and checked which one gave the highest score.
Visualization
Lastly, I plotted a line chart using matplotlib to visually compare the accuracy scores across different k values to see how the model performed.
