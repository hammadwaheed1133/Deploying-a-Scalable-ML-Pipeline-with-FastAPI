import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained model using precision, recall, and F1 score.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return predictions.
    """
    return model.predict(X)


def save_model(model, path):
    """
    Serializes model to a file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """
    Loads a model from a pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model
):
    """
    Computes the model metrics on a slice of the data specified by column name
    and value.
    """
    data_slice = data[data[column_name] == slice_value]

    X_slice, y_slice, _, _ = process_data(
        X=data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
