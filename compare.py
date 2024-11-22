# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:/Users/ongyu/Downloads/GoldPrice.csv")

# ---- 1. Before Preprocessing ----
print("First 5 rows of data (Before Preprocessing):")
print(data.head(), "\n")

print("Data Information (Before Preprocessing):")
print(data.info(), "\n")

print("Missing Values (Before Preprocessing):")
print(data.isnull().sum(), "\n")

# ---- 2. Data Preprocessing ----
# Remove commas from the dataset (replace with empty strings)
data = data.replace({',': ''}, regex=True)

# Replace 'K' and 'M' in 'Vol.' column with their numeric equivalents
data['Vol.'] = data['Vol.'].str.replace('K', 'e3').str.replace('M', 'e6')

# Convert 'Vol.' to numeric and show result
data['Vol.'] = pd.to_numeric(data['Vol.'], errors='coerce')

# Remove percentage symbol from 'Change %' column and convert to numeric
data['Change %'] = data['Change %'].str.replace('%', '')
data['Change %'] = pd.to_numeric(data['Change %'], errors='coerce')

# Convert 'Date' to datetime format, then drop it
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Drop unused column
data.drop(columns=['Date'], inplace=True)

# Replace missing values with the column mean
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Show dataset information after handling missing values
print("Missing Values After Replacing with Mean:")
print(data.isnull().sum(), "\n")

# Print summary statistics of the data after preprocessing
print("Summary Statistics After Preprocessing:")
print(data.describe(), "\n")

# ---- 3. Model Training ----
# Define the target (Price) and feature variables (all other columns)
X = data.drop(columns=['Price'])
y = data['Price']

# Ensure that X and y are numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 3. Model Training ----
# a) Linear Regression
# Training the Linear Regression model
linear = LinearRegression()
linear.fit(X_train, y_train)

# Making predictions on the test set
y_pred_linear = linear.predict(X_test)

# Evaluating the Linear Regression model
MAE_linear = mean_absolute_error(y_test, y_pred_linear)
RMSE_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
R2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression: ")
print(f"MAE: {MAE_linear}, RMSE: {RMSE_linear}, R²: {R2_linear}\n")

# b) Random Forest Regression
# Training the Random Forest model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Making predictions on the test set
y_pred_rf = random_forest.predict(X_test)

# Evaluating the Random Forest model
MAE_rf = mean_absolute_error(y_test, y_pred_rf)
RMSE_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
R2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regression: ")
print(f"MAE: {MAE_rf}, RMSE: {RMSE_rf}, R²: {R2_rf}\n")

# ---- 4. Fine Tuning Model ----
# Define the hyperparameters grid to search
rf_param_grid = {
    'n_estimators': [100, 200, 300], # Number of trees in the forest (increase accuracy)
    'max_depth': [None, 10, 20, 30], # Maximum depth of each tree (ability to catch pattern)
    'min_samples_split': [2, 5, 10]  # Minimum number of samples required to split a node (prevent overfitting)
}

# Perform GridSearchCV for Random Forest
random_forest_grid = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                              param_grid=rf_param_grid,
                              cv=5, n_jobs=-1, verbose=2)

# Fit the grid search
random_forest_grid.fit(X_train, y_train)

# Get the best parameters from grid search
best_random_forest = random_forest_grid.best_estimator_

# Make predictions using the tuned Random Forest model
y_pred_rf_tuned = best_random_forest.predict(X_test)

# Evaluate the tuned Random Forest model
MAE_rf_tuned = mean_absolute_error(y_test, y_pred_rf_tuned)
RMSE_rf_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned))
R2_rf_tuned = r2_score(y_test, y_pred_rf_tuned)

print(f"Tuned Random Forest Regression - MAE: {MAE_rf_tuned}, RMSE: {RMSE_rf_tuned}, R²: {R2_rf_tuned}\n")

# ---- 5. Comparison of Linear Regression and Random Forest ----
print("Comparison of Models:")
print(f"Linear Regression - MAE: {MAE_linear}, RMSE: {RMSE_linear}, R²: {R2_linear}")
print(f"Random Forest (Tuned) - MAE: {MAE_rf_tuned}, RMSE: {RMSE_rf_tuned}, R²: {R2_rf_tuned}")

# ---- 6. Visualization ---- 
# a) Residual Plot for Linear Regression
residuals_linear = y_test - y_pred_linear

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred_linear, y=residuals_linear)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals for Linear Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# b) Prediction vs Actual for Linear Regression
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_linear)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Linear Regression: Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# c) Residual Plot for Tuned Random Forest
residuals_rf_tuned = y_test - y_pred_rf_tuned

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred_rf_tuned, y=residuals_rf_tuned)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals for Tuned Random Forest Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# d) Prediction vs Actual for Tuned Random Forest
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_rf_tuned)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Tuned Random Forest Regression: Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# e) Feature Importance Plot for Tuned Random Forest
feature_importances = best_random_forest.feature_importances_
features = X.columns
importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importance in Tuned Random Forest Regression')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
