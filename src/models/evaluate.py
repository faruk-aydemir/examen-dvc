import os
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    os.makedirs("metrics", exist_ok=True)

    model = joblib.load("models/gbr_model.pkl")

    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze()

    predictions = model.predict(X_test)

    pred_df = pd.DataFrame({"prediction": predictions})
    pred_df.to_csv("data/predictions.csv", index=False)

    scores = {
        "mse": mean_squared_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions)
    }

    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("Evaluation completed.")
    print(scores)


if __name__ == "__main__":
    main()