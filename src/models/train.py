import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    best_params = joblib.load("models/best_params.pkl")

    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze()

    model = GradientBoostingRegressor(
        random_state=params["train"]["random_state"],
        **best_params
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/gbr_model.pkl")

    print("Training completed.")


if __name__ == "__main__":
    main()