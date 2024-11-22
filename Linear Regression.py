# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with the path to your file if necessary)
data = pd.read_csv("C:/Users/ongyu/Downloads/GoldPrice.csv")

print("Missing values in dataset: ")
print(data.isna().sum())

print("First 10 row of data before preprocessing:")
print(data.head(10))

# Pre-process the dataset
# 1. Remove commas from numeric columns and convert to float
data = data.replace({',': ''}, regex=True)

# 2. Convert 'Vol.' column to numeric (removing 'K' and 'M')
data['Vol.'] = data['Vol.'].str.replace('K', 'e3').str.replace('M', 'e6')
data['Vol.'] = pd.to_numeric(data['Vol.'], errors='coerce')

# 3. Convert 'Change %' column to numeric (removing '%' and making it a decimal)
data['Change %'] = data['Change %'].str.replace('%', '')
data['Change %'] = pd.to_numeric(data['Change %'], errors='coerce')

# 4. Convert 'Date' column to datetime (if needed)
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Handle missing data: Fill missing values only in numeric columns with the mean
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Drop the 'Date' column (since it's not needed for modeling)
data.drop(columns=['Date'], inplace=True)

# Now the dataset is ready for modeling
print("First 10 row of data after preprocessing:")
print(data.head(10))

# Linear Regression Model
# Define the target (Price) and feature variables (all other columns)
X = data.drop(columns=['Price'])
y = data['Price']

# Ensure that X and y are numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Linear Regression Model ###
# Training the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred_linear = linear_model.predict(X_test)

# Evaluating the Linear Regression model
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

# Print the results for Linear Regression
print(f"Linear Regression - MAE: {mae_linear}, RMSE: {rmse_linear}, R2 score: {r2_linear}")

# Plot Actual vs. Predicted values for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, color='blue', label='Linear Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted (Linear Regression)')
plt.legend()
plt.show()

# Residuals for Linear Regression
# Ensure y_test and y_pred_linear are numeric before subtracting
y_test = pd.to_numeric(y_test, errors='coerce')
y_pred_linear = pd.to_numeric(y_pred_linear, errors='coerce')

residuals_linear = y_test - y_pred_linear

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_linear, residuals_linear, color='blue', label='Linear Regression Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot (Linear Regression)')
plt.legend()
plt.show()

# Plot distribution of Actual vs. Predicted values for Linear Regression
plt.figure(figsize=(8, 6))
sns.histplot(y_test, kde=True, color='blue', label='Actual')
sns.histplot(y_pred_linear, kde=True, color='green', label='Predicted')
plt.title('Distribution of Actual vs Predicted Prices (Linear Regression)')
plt.legend()
plt.show()


