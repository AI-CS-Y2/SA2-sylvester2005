import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('Mobile phone price.csv')

# Show basic information about the dataset
print("\nFirst 5 rows of the dataset:\n")
print(data.head())

# Preprocessing
# Encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Visualize the distribution of phone prices
plt.figure(figsize=(10, 6))
sns.histplot(data['Price ($)'], bins=30, kde=True, color='blue')
plt.title("Distribution of Phone Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Linear Regression: Predict phone price
X = data.drop(columns=['Price ($)', 'Brand', 'Model'])  # Dropping 'Brand' and 'Model' columns along with 'Price ($)'
y = data['Price ($)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predict and evaluate Linear Regression
y_pred_linear = linear_model.predict(X_test_scaled)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression Evaluation:")
print(f"  Mean Squared Error (MSE): {mse_linear:.2f}")
print(f"  R-squared (R²): {r2_linear * 100:.2f}%")

# Logistic Regression: Classify price into 'low' (0) or 'high' (1)
# Create a binary price category
threshold = data['Price ($)'].median()
data['price_category'] = np.where(data['Price ($)'] > threshold, 1, 0)

X_class = data.drop(columns=['Price ($)', 'price_category', 'Brand', 'Model'])
y_class = data['price_category']

# Train-test split for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=2000, solver='lbfgs')
logistic_model.fit(X_train_class, y_train_class)

# Predict and evaluate Logistic Regression
y_pred_class = logistic_model.predict(X_test_class)
accuracy_logistic = accuracy_score(y_test_class, y_pred_class) * 100  # Convert to percentage
classification_report_logistic = classification_report(y_test_class, y_pred_class)

print("\nLogistic Regression Evaluation:")
print(f"  Accuracy: {accuracy_logistic:.2f}%")
print("  Classification Report:\n", classification_report_logistic)

# Visualization: Confusion Matrix for Logistic Regression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_matrix = confusion_matrix(y_test_class, y_pred_class)
cmd = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Low", "High"])
cmd.plot(cmap=plt.cm.Blues)
plt.title("Logistic Regression: Confusion Matrix")
plt.show()

# Compare models using Cross-Validation
logistic_cv_score = cross_val_score(logistic_model, X_class, y_class, cv=5, scoring='accuracy').mean()
linear_cv_score = cross_val_score(linear_model, X, y, cv=5, scoring='r2').mean()

print(f"\nLogistic Regression Cross-Validation Accuracy: {logistic_cv_score * 100:.2f}%")
print(f"Linear Regression Cross-Validation R²: {linear_cv_score * 100:.2f}%")

# Model Tuning: Hyperparameter tuning for Logistic Regression using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
grid_search = GridSearchCV(LogisticRegression(max_iter=2000), param_grid, cv=5)
grid_search.fit(X_train_class, y_train_class)

best_params = grid_search.best_params_
print("\nBest Hyperparameters for Logistic Regression:", best_params)
best_logistic_model = grid_search.best_estimator_

# Evaluate tuned Logistic Regression
y_pred_best_class = best_logistic_model.predict(X_test_class)
accuracy_best_logistic = accuracy_score(y_test_class, y_pred_best_class) * 100
print(f"Tuned Logistic Regression Accuracy: {accuracy_best_logistic:.2f}%")

# Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_class, y_train_class)

# Predict and evaluate Random Forest
y_pred_rf_class = random_forest_model.predict(X_test_class)
accuracy_rf = accuracy_score(y_test_class, y_pred_rf_class) * 100
print(f"Random Forest Classifier Accuracy: {accuracy_rf:.2f}%")

# Final Comparison
print("\nFinal Model Comparison:")
print(f"Logistic Regression Accuracy: {accuracy_best_logistic:.2f}%")
print(f"Random Forest Accuracy: {accuracy_rf:.2f}%")
