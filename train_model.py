import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("--- Starting Model Training ---")

# --- 1. Load Data ---
# Load the dataset from the CSV file
try:
    df = pd.read_csv('C:/Users/shita/OneDrive/Desktop/streamlit_price_predictor/Phone.csv.csv')
    print("Successfully loaded phone.csv")
except FileNotFoundError:
    print("Error: phone.csv not found. Please make sure it's in the same folder.")
    exit() # Exit the script if the file is not found

# --- 2. Prepare Data ---
# Separate the features (X) from the target variable (y)
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into training and testing sets.")

# --- 3. Scale Features ---
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler.")

# --- 4. Train the Model ---
# We use a RandomForestClassifier, which is powerful for this type of problem.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 5. Evaluate the Model ---
# Check the model's performance on unseen test data
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy:.2f}")

# --- 6. Save the Model and Scaler ---
# Save the trained model and scaler objects to disk for our app to use
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler have been saved successfully.")

print("--- Model Training Script Finished ---")