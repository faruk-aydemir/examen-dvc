import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    input_dir = "data/processed_data"
    output_dir = "data/processed_data"
    model_dir = "models"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)

    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    print("Normalization completed.")


if __name__ == "__main__":
    main()