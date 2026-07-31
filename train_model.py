import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("=" * 60)
print("Heart Disease Prediction - Model Training")
print("=" * 60)

# ========================
# TASK 1: Data Understanding and Preprocessing
# ========================
print("\n--- TASK 1: Data Understanding and Preprocessing ---\n")

# Download dataset using kagglehub
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print(f"Path to dataset files: {path}")

# Load the dataset
df = pd.read_csv(f"{path}/heart.csv")
print(f"\nDataset Shape: {df.shape}\n")

# Display first five records
print("First 5 records:")
print(df.head())

# Dataset info
print(f"\nDataset Info:")
print(df.info())

# Statistical summary
print(f"\nStatistical Summary:")
print(df.describe())

# Identify numerical features and target variable
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
target = 'target'

print(f"\nNumerical Features ({len(numerical_features)}): {numerical_features}")
print(f"Target Variable: {target}")
print(f"Target Distribution:")
print(df[target].value_counts())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Split the dataset into 80% training and 20% testing
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# ========================
# TASK 2: Model Development
# ========================
print("\n--- TASK 2: Model Development ---\n")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Feature importance
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

# Save the trained model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel saved as 'model.pkl'")
print("Scaler saved as 'scaler.pkl'")

# ========================
# TASK 5: Conclusion
# ========================
print("\n--- TASK 5: Conclusion ---\n")
print("""This project successfully developed a Random Forest classification model to predict 
heart disease risk based on clinical parameters. The model achieved an accuracy of 
{:.2f}%, demonstrating strong predictive performance for this binary classification task. 

Key challenges faced during deployment included ensuring proper data preprocessing 
consistency between training and inference, and configuring the Flask API for cloud 
deployment on Render. 

MLOps practices are crucial for machine learning projects as they bridge the gap between 
model development and production deployment, ensuring reproducibility, monitoring, and 
continuous improvement of ML systems.""".format(accuracy * 100))

print("\n" + "=" * 60)
print("Model Training Complete!")
print("=" * 60)
