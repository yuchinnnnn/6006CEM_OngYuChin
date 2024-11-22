import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Define Data Cleaning Function
def clean_data(data):
    # Remove rows with '?' in any categorical column
    data = data[data.select_dtypes(include=['object']).apply(lambda x: x != '?').all(axis=1)]
    
    # Strip extra spaces from column names
    data.columns = data.columns.str.strip()
    
    # Remove classes with fewer than 2 samples in 'native-country'
    if 'native-country' in data.columns:
        data = data[data['native-country'].map(data['native-country'].value_counts()) > 1]
    
    return data

# Step 2: Define Encoding Function
def encode_categorical_columns(data):
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le  # Store encoders
    return data, label_encoders

# Step 3: Define Train and Evaluate Function
def train_and_evaluate(data, target_column, classifier=RandomForestClassifier(random_state=42)):
    print(f"\nStarting Training for Target: {target_column}")
    try:
        # Prepare features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train classifier
        classifier.fit(X_train, y_train)

        # Evaluate model
        train_accuracy = classifier.score(X_train, y_train)
        test_accuracy = classifier.score(X_test, y_test)
        y_pred = classifier.predict(X_test)

        # Display results
        print(f"\n{target_column.upper()} CLASSIFICATION")
        print(f"Training Accuracy: {train_accuracy:.2f}")
        print(f"Testing Accuracy: {test_accuracy:.2f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("-" * 50)
    except ValueError as e:
        print(f"Error during training for target '{target_column}': {e}")

# Main Script
if __name__ == "__main__":
    # Load dataset
    file_path = "C:/Users/ongyu/Downloads/income_evaluation.csv"
    data = pd.read_csv(file_path)
    
    print(data.head())

    # Print missing values
    print("Number of missing values in each column:")
    print(data.isna().sum())

    # Print initial class distribution for 'native-country'
    if 'native-country' in data.columns:
        print("\nOriginal 'native-country' class distribution:")
        print(data['native-country'].value_counts())

    # Clean data
    data = clean_data(data)

    # Print filtered class distribution for 'native-country'
    if 'native-country' in data.columns:
        print("\nFiltered 'native-country' class distribution:")
        print(data['native-country'].value_counts())

    # Encode categorical columns
    data, label_encoders = encode_categorical_columns(data)

    # Verify cleaned and encoded dataset
    print("\nCleaned and Encoded Dataset Overview:")
    print(data.info())

    # Define classification tasks
    classification_tasks = [
        'workclass', 'sex', 'marital-status', 'race', 'education', 'native-country', 'occupation'
    ]

    # Train and evaluate models for each classification task
    for target in classification_tasks:
        if target in data.columns:
            train_and_evaluate(data, target)
        else:
            print(f"Warning: Column '{target}' not found in the dataset.")
