import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

FOLDER = Path(__file__).resolve().parents[1]


def test_01():

    dataset = pd.read_csv(FOLDER / "data" / "auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")
    y_true = dataset.pop("MPG")

    with open(FOLDER / "submission" / "mlp.pkl", "rb") as file:
        mlp = pickle.load(file)

    with open(FOLDER / "submission" / "features_scaler.pkl", "rb") as file:
        features_scaler = pickle.load(file)

    feature_names = getattr(features_scaler, "feature_names_in_", None)
    if feature_names is not None:
        dataset = dataset.reindex(columns=feature_names, fill_value=0)

    standarized_dataset = features_scaler.transform(dataset)
    y_pred = mlp.predict(standarized_dataset)

    mse = mean_squared_error(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert mse < 7.745
