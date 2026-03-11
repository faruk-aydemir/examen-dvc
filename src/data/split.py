import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    input_path = "data/raw_data/raw.csv"
    output_dir = "data/processed_data"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    # Keep only numeric columns
    df = df.select_dtypes(include=["number"])

    # Last numeric column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_frame(name="silica_concentrate").to_csv(
        os.path.join(output_dir, "y_train.csv"), index=False
    )
    y_test.to_frame(name="silica_concentrate").to_csv(
        os.path.join(output_dir, "y_test.csv"), index=False
    )

    print("Data split completed.")


if __name__ == "__main__":
    main()