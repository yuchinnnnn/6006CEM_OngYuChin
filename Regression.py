# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Data
data = pd.read_csv("C:/Users/ongyu/Downloads/GoldPrice.csv")

# Pre-process the dataset
data = data.replace({',': ''}, regex=True)
data['Vol.'] = data['Vol.'].str.replace('K', 'e3').str.replace('M', 'e6')
data['Vol.'] = pd.to_numeric(data['Vol.'], errors='coerce')
data['Change %'] = data['Change %'].str.replace('%', '')
data['Change %'] = pd.to_numeric(data['Change %'], errors='coerce')
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Handle missing data
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
data.drop(columns=['Date'], inplace=True)

# Define the target (Price) and feature variables (all other columns)
X = data.drop(columns=['Price'])
y = data['Price']

# Ensure that X and y are numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Random Forest Regression Model ###
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions using Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluating the Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Print the results for Random Forest
print(f"Random Forest Regression - MAE: {mae_rf}, RMSE: {rmse_rf}, R2 score: {r2_rf}")

# Plot Actual vs. Predicted values for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, color='green', label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted (Random Forest)')
plt.legend()
plt.show()

# Residuals for Random Forest
residuals_rf = y_test - y_pred_rf

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_rf, residuals_rf, color='green', label='Random Forest Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot (Random Forest)')
plt.legend()
plt.show()

# Plot distribution of Actual vs. Predicted values for Random Forest
plt.figure(figsize=(8, 6))
sns.histplot(y_test, kde=True, color='blue', label='Actual')
sns.histplot(y_pred_rf, kde=True, color='orange', label='Predicted')
plt.title('Distribution of Actual vs Predicted Prices (Random Forest)')
plt.legend()
plt.show()

# Feature Importance Visualization
feature_importances = rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances from Random Forest')
plt.show()
