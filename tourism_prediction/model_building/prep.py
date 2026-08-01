import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_prediction", "data", "tourism.csv")
TARGET = "ProdTaken"
DROP_COLS = ["CustomerID"]


def main():
    df = pd.read_csv(DATA_PATH)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)
    print("Prep done:", Xtrain.shape, Xtest.shape)


if __name__ == "__main__":
    main()
