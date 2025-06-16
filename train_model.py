# train_model.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from model import fit_model, evaluate_model, generate_predictions

# Load census data
csv_path = os.path.join("data", "census.csv")
df = pd.read_csv(csv_path)

# Feature/label separation
X = df.drop("salary", axis=1)
y = df["salary"].apply(lambda val: 1 if val == ">50K" else 0)

# Divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the classifier
classifier = fit_model(X_train, y_train)

# Generate predictions and evaluate
predictions = generate_predictions(classifier, X_test)
precision, recall, f1 = evaluate_model(y_test, predictions)

print("[INFO] Evaluation results:")
print(f"  - Precision: {precision:.3f}")
print(f"  - Recall:    {recall:.3f}")
print(f"  - F1 Score:  {f1:.3f}")

# Save trained model to disk
output_path = os.path.join("model", "trained_classifier.pkl")
joblib.dump(classifier, output_path)
print(f"[INFO] Model saved at: {output_path}")
