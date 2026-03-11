import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    cv = params["grid_search"]["cv"]
    scoring = params["grid_search"]["scoring"]

    param_grid = {
        "n_estimators": params["model"]["n_estimators"],
        "learning_rate": params["model"]["learning_rate"],
        "max_depth": params["model"]["max_depth"],
    }

    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze()

    model = GradientBoostingRegressor(random_state=params["train"]["random_state"])

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_params_, "models/best_params.pkl")

    print("Best params:", grid.best_params_)


if __name__ == "__main__":
    main()