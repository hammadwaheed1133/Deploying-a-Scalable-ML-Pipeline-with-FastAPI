import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
import pandas as pd

# Load a small sample dataframe for testing
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        "age": [39, 50],
        "workclass": ["State-gov", "Self-emp-not-inc"],
        "fnlgt": [77516, 83311],
        "education": ["Bachelors", "Bachelors"],
        "education-num": [13, 13],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [2174, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })
    return data

def test_train_model(sample_data):
    cat_features = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]
    X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression)

def test_inference_output_type(sample_data):
    cat_features = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]
    X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)

def test_compute_model_metrics_values():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
