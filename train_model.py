import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census.csv data
project_path = os.getcwd()  # Set project path dynamically
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)

# Split the data into train and test sets
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Categorical features to use
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

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Save the model and encoder
os.makedirs(os.path.join(project_path, "model"), exist_ok=True)
save_model(model, os.path.join(project_path, "model", "model.pkl"))
save_model(encoder, os.path.join(project_path, "model", "encoder.pkl"))
save_model(lb, os.path.join(project_path, "model", "label_binarizer.pkl"))

# Load model (optional redundancy, for test/demo)
model = load_model(os.path.join(project_path, "model", "model.pkl"))

# Run inference
preds = inference(model, X_test)

# Compute and print metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute metrics on slices and save to file
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model
            )
            f.write(f"{col}: {slicevalue}, Count: {count:,}\n")
            f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n\n")
save_model(model, "model/model.pkl")
save_model(encoder, "model/encoder.pkl")
save_model(lb, "model/lb.pkl") 