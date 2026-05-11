import pandas as pd
import numpy as np

DATA_PATH = "data/healthcare-dataset-stroke-data.csv"

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()

    # BMI has missing values in this dataset, so fill them with the median BMI.
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Drop id because it does not help predict or analyze stroke risk.
    df = df.drop(columns=["id"])

    return df


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    print("Data loaded and cleaned successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()