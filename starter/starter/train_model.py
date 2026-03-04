# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
import pickle
import os
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
# Remove spaces from column names
data.columns = data.columns.str.strip()
# Remove spaces from string values
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].str.strip()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Create model directory if it doesn't exist
os.makedirs("../model", exist_ok=True)

# Save the model
with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the encoder
with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Save the label binarizer
with open("../model/lb.pkl", "wb") as f:
    pickle.dump(lb, f)

# Evaluate the model on test data
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("Model Performance on Test Set:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {fbeta:.4f}")

# Compute performance on slices of the data
def compute_slice_metrics(data, categorical_features, model, encoder, lb, output_file="../model/slice_output.txt"):
    """
    Compute and save model performance on slices of categorical features.
    
    Inputs
    ------
    data : pd.DataFrame
        Test data.
    categorical_features : list
        List of categorical feature names.
    model : RandomForestClassifier
        Trained model.
    encoder : OneHotEncoder
        Fitted encoder.
    lb : LabelBinarizer
        Fitted label binarizer.
    output_file : str
        Path to output file.
    """
    with open(output_file, "w") as f:
        for feature in categorical_features:
            for value in data[feature].unique():
                # Filter data for this slice
                slice_data = data[data[feature] == value]
                
                if len(slice_data) > 0:
                    # Process the slice
                    X_slice, y_slice, _, _ = process_data(
                        slice_data, 
                        categorical_features=categorical_features, 
                        label="salary", 
                        training=False, 
                        encoder=encoder, 
                        lb=lb
                    )
                    
                    # Get predictions
                    preds = inference(model, X_slice)
                    
                    # Compute metrics
                    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
                    
                    # Write to file
                    f.write(f"{feature} = {value}\n")
                    f.write(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fbeta:.4f}\n")
                    f.write(f"  Sample size: {len(slice_data)}\n\n")

# Compute performance on slices
print("\nComputing performance on slices of categorical features...")
compute_slice_metrics(test, cat_features, model, encoder, lb)
print("Slice performance saved to ../model/slice_output.txt")
