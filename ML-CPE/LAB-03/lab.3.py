import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv('data (2).csv')

print(df.head())
df = df.replace('?', np.nan)
df = df.astype(float)

imputer = SimpleImputer(strategy='mean')
df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("จำนวนข้อมูลทั้งหมด:", df_clean.shape)
df_clean.head()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


X_reg = df_clean.drop('age', axis=1)
y_reg = df_clean['age']


X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)


scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)


pca_r = PCA(n_components=5)
X_train_r_pca = pca_r.fit_transform(X_train_r_scaled)
X_test_r_pca = pca_r.transform(X_test_r_scaled)


reg_model = LinearRegression()
reg_model.fit(X_train_r_pca, y_train_r)

y_pred_r = reg_model.predict(X_test_r_pca)
print("--- ผลการประเมิน Linear Regression (ทำนายอายุ) ---")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_r, y_pred_r):.2f} ปี")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test_r, y_pred_r)):.2f} ปี")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


X_clf = df_clean.drop('sex', axis=1)
y_clf = df_clean['sex']


X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)


scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

pca_c = PCA(n_components=5)
X_train_c_pca = pca_c.fit_transform(X_train_c_scaled)
X_test_c_pca = pca_c.transform(X_test_c_scaled)


clf_model = LogisticRegression()
clf_model.fit(X_train_c_pca, y_train_c)
y_pred_c = clf_model.predict(X_test_c_pca)
y_prob_c = clf_model.predict_proba(X_test_c_pca)[:, 1] 

print("--- ผลการประเมิน Classification (จำแนกเพศ) ---")
print(f"Accuracy:  {accuracy_score(y_test_c, y_pred_c):.2f}")
print(f"Precision: {precision_score(y_test_c, y_pred_c):.2f}")
print(f"Recall:    {recall_score(y_test_c, y_pred_c):.2f}")
print(f"F1-score:  {f1_score(y_test_c, y_pred_c):.2f}")
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_c))


fpr, tpr, thresholds = roc_curve(y_test_c, y_prob_c)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Gender Classification')
plt.legend(loc="lower right")
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test_c, y_pred_c)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0 (Female)', 'Predicted 1 (Male)'],
            yticklabels=['Actual 0 (Female)', 'Actual 1 (Male)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Gender Classification')
plt.show()


pca_2d = PCA(n_components=2)
X_train_c_2d = pca_2d.fit_transform(X_train_c_scaled)

clf_model_2d = LogisticRegression()
clf_model_2d.fit(X_train_c_2d, y_train_c)

x_min, x_max = X_train_c_2d[:, 0].min() - 1, X_train_c_2d[:, 0].max() + 1
y_min, y_max = X_train_c_2d[:, 1].min() - 1, X_train_c_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
plt.scatter(X_train_c_2d[:, 0], X_train_c_2d[:, 1], c=y_train_c, edgecolor='k', cmap=plt.cm.coolwarm)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Boundary - Logistic Regression (PCA 2D)')
plt.show()
from sklearn.metrics import r2_score

print("--- LAB 3: Model Comparison (การเปรียบเทียบโมเดล Regression) ---")

y_train_pred_r = reg_model.predict(X_train_r_pca)
y_test_pred_r = reg_model.predict(X_test_r_pca)

train_rmse = np.sqrt(mean_squared_error(y_train_r, y_train_pred_r))
test_rmse = np.sqrt(mean_squared_error(y_test_r, y_test_pred_r))
train_r2 = r2_score(y_train_r, y_train_pred_r)
test_r2 = r2_score(y_test_r, y_test_pred_r)

print(f"Training RMSE: {train_rmse:.2f} | R2 Score: {train_r2:.2f}")
print(f"Testing  RMSE: {test_rmse:.2f} | R2 Score: {test_r2:.2f}")

X_simple_train = X_train_r_scaled[:, [0]] # เลือกมา 1 ฟีเจอร์
X_simple_test = X_test_r_scaled[:, [0]]

simple_reg = LinearRegression()
simple_reg.fit(X_simple_train, y_train_r)
y_simple_pred = simple_reg.predict(X_simple_test)

simple_rmse = np.sqrt(mean_squared_error(y_test_r, y_simple_pred))
simple_r2 = r2_score(y_test_r, y_simple_pred)

print("\n--- ผลเปรียบเทียบ Simple vs Multiple Linear Regression ---")
print(f"Simple Linear Regression   - RMSE: {simple_rmse:.2f} | R2 Score: {simple_r2:.2f}")
print(f"Multiple Linear Reg (PCA)  - RMSE: {test_rmse:.2f} | R2 Score: {test_r2:.2f}")