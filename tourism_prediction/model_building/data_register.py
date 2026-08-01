import os
import pandas as pd

DATA_DIR = os.path.join("tourism_prediction", "data")
DATA_PATH = os.path.join(DATA_DIR, "tourism.csv")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH)
    assert "ProdTaken" in df.columns, "Target column 'ProdTaken' is missing."
    assert len(df) > 0, "Dataset is empty."
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    df.to_csv(DATA_PATH, index=False)
    print("Dataset registered:", DATA_PATH, df.shape)


if __name__ == "__main__":
    main()
