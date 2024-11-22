import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Load dataset
file_path = "C:/Users/ongyu/Downloads/income_evaluation.csv"
df = pd.read_csv(file_path)

print("Before preprocessing: ")
print("First 5 rows of data: ")
print(df.head(20))
print("Data info: ")
print(df.info())

# Data Preprocessing
df = df.rename(columns = {' workclass':'workclass',' fnlwgt':'fnlwgt',
                          ' education':'education',
                          ' education-num':'education-num',
                          ' marital-status':'marital-status',
                          ' occupation':'occupation',
                          ' relationship':'relationship',
                          ' race':'race',' sex':'sex',
                          ' capital-gain':'capital-gain',
                          ' capital-loss':'capital-loss',
                          ' hours-per-week':'hours-per-week',
                          ' native-country':'country',' income':'income'})

df = df.drop('education', axis=1)

# Clean up spaces in column names and values
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Filter data to only include "United-States"
df = df[df['country'] == 'United-States']
df = df.drop('country', axis=1)

# Encode target variable 'income' and 'sex' to numerical
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Handle categorical variables by replacing certain values
df['workclass'] = df['workclass'].replace(['Without-pay', 'Never-worked'], 'Never-worked')
df['workclass'] = df['workclass'].replace(['Local-gov', 'Self-emp-not-inc', 'State-gov'], 
                                          'State-gov,local-gov,self-emp-not-inc')
df['marital-status'] = df['marital-status'].replace(['Divorced', 'Married-spouse-absent', 
                                                     'Never-married', 'Separated', 'Widowed'], 'Single')
df['marital-status'] = df['marital-status'].replace(['Married-AF-spouse', 'Married-civ-spouse'], 'Married')
df['occupation'] = df['occupation'].replace(['?', 'Adm-clerical', 'Armed-Forces', 'Machine-op-inspct', 'Farming-fishing'], 
                                            'Adm-clerical,Armed-Forces,Machine-op-inspct,?,Farming-fishing')
df['relationship'] = df['relationship'].replace(['Husband', 'Wife'], 'Wife,husband')
df['relationship'] = df['relationship'].replace(['Not-in-family', 'Other-relative', 'Own-child', 
                                                 'Unmarried'], 'Unmarried')

# Drop outliers or unwanted features
df = df.drop(['fnlwgt', 'capital-gain', 'capital-loss', 'relationship'], axis=1)

# Apply One-Hot Encoding to categorical variables
df = pd.get_dummies(df, columns=['workclass', 'marital-status', 'occupation', 'race', 'sex'], drop_first=True)
df = df.astype(int)

# Handling missing values using SMOTE
x = df.drop('income', axis=1)
y = df['income']

# Ensure that x and y are not empty
if x.empty or y.empty:
    print("Error: The feature matrix or target vector is empty.")
else:
    # Proceed with SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=0)
    x, y = smote.fit_resample(x, y)
    
print("After preprocessing: ")
print("First 5 row of data: ")
print(df.head(20))
print("Data info: ")
print(df.info())

# Check data shape and verify that x and y are not empty
print(f"Shape of df after cleaning: {df.shape}")
print(f"Shape of x: {x.shape}")
print(f"Shape of y: {y.shape}")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

# --- Original Models --- #

# Decision Tree Model
decision_tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
decision_tree_model.fit(x_train, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(x_train, y_train)

# Make predictions
y_pred_dt = decision_tree_model.predict(x_test)
y_pred_rf = rf_model.predict(x_test)

# Classification Report
print("Original Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Original Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
cmd_dt = ConfusionMatrixDisplay(cm_dt, display_labels=decision_tree_model.classes_)
cmd_dt.plot(cmap='Blues')
plt.title('Original Decision Tree Confusion Matrix')
plt.show()

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
cmd_rf = ConfusionMatrixDisplay(cm_rf, display_labels=rf_model.classes_)
cmd_rf.plot(cmap='Blues')
plt.title('Original Random Forest Confusion Matrix')
plt.show()

# Compare Model Performance
dt_accuracy = accuracy_score(y_test, y_pred_dt)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"Original Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Original Random Forest Accuracy: {rf_accuracy:.4f}")

# --- Fine-Tuned Models --- #

# Fine-Tuning Decision Tree using GridSearchCV
dt_param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=dt_param_grid, cv=5, n_jobs=-1, verbose=1)
dt_grid_search.fit(x_train, y_train)

# Best parameters and score for Decision Tree
print("Fine-Tuned Decision Tree Best Parameters:", dt_grid_search.best_params_)
print("Fine-Tuned Decision Tree Best Score:", dt_grid_search.best_score_)

# Predict with the best Decision Tree model
y_pred_dt_tuned = dt_grid_search.best_estimator_.predict(x_test)

# Classification report and confusion matrix for Fine-Tuned Decision Tree
print("Fine-Tuned Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt_tuned))
cm_dt_tuned = confusion_matrix(y_test, y_pred_dt_tuned)
cmd_dt_tuned = ConfusionMatrixDisplay(cm_dt_tuned, display_labels=dt_grid_search.best_estimator_.classes_)
cmd_dt_tuned.plot(cmap='Blues')
plt.title('Fine-Tuned Decision Tree Confusion Matrix')
plt.show()

# Fine-Tuning Random Forest using GridSearchCV
rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=1)
rf_grid_search.fit(x_train, y_train)

# Best parameters and score for Random Forest
print("Fine-Tuned Random Forest Best Parameters:", rf_grid_search.best_params_)
print("Fine-Tuned Random Forest Best Score:", rf_grid_search.best_score_)

# Predict with the best Random Forest model
y_pred_rf_tuned = rf_grid_search.best_estimator_.predict(x_test)

# Classification report and confusion matrix for Fine-Tuned Random Forest
print("Fine-Tuned Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf_tuned))
cm_rf_tuned = confusion_matrix(y_test, y_pred_rf_tuned)
cmd_rf_tuned = ConfusionMatrixDisplay(cm_rf_tuned, display_labels=rf_grid_search.best_estimator_.classes_)
cmd_rf_tuned.plot(cmap='Blues')
plt.title('Fine-Tuned Random Forest Confusion Matrix')
plt.show()

# Compare Model Performance for Fine-Tuned Models
dt_accuracy_tuned = accuracy_score(y_test, y_pred_dt_tuned)
rf_accuracy_tuned = accuracy_score(y_test, y_pred_rf_tuned)

print(f"Fine-Tuned Decision Tree Accuracy: {dt_accuracy_tuned:.4f}")
print(f"Fine-Tuned Random Forest Accuracy: {rf_accuracy_tuned:.4f}")

# --- Additional Graphs and Plots --- #

# Plot Accuracy vs Epoch (using the best tuned models)
# Assuming you are interested in tracking accuracy during training
# For simplicity, you can plot training/validation accuracy (if you had history data)
# Placeholder example for Random Forest and Decision Tree:
# If you had training history data, you could plot them like this:

# Accuracy vs Epoch Plot (for fine-tuned models, assuming you had a training history):
# plt.plot(history_dt.history['accuracy'], label='Decision Tree Training Accuracy')
# plt.plot(history_rf.history['accuracy'], label='Random Forest Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Accuracy vs Epochs')
# plt.show()

# ROC Curve for Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)
plt.plot(fpr_dt, tpr_dt, label='ROC curve (area = %0.2f)' % roc_auc_dt)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend(loc='lower right')
plt.show()

# ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve for Decision Tree
precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_pred_dt)
plt.plot(recall_dt, precision_dt, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Decision Tree')
plt.legend()
plt.show()

# Precision-Recall Curve for Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf)
plt.plot(recall_rf, precision_rf, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Random Forest')
plt.legend()
plt.show()