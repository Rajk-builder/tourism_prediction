import os
import pandas as pd
import joblib
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
import xgboost as xgb
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("tourism-package-prediction")

DEPLOY_DIR = os.path.join("tourism_prediction", "deployment")
MODEL_PATH = os.path.join(DEPLOY_DIR, "best_tourism_model.joblib")


def main():
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")

    cat_cols = Xtrain.select_dtypes(include=["object"]).columns.tolist()
    num_cols = Xtrain.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), cat_cols),
        (StandardScaler(), num_cols),
        remainder="drop",
    )
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
    pipe = make_pipeline(preprocessor, model)

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.1, 0.05],
    }
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1)

    with mlflow.start_run():
        grid.fit(Xtrain, ytrain)
        best = grid.best_estimator_
        preds = best.predict(Xtest)
        metrics = {
            "accuracy": accuracy_score(ytest, preds),
            "precision": precision_score(ytest, preds, zero_division=0),
            "recall": recall_score(ytest, preds, zero_division=0),
            "f1": f1_score(ytest, preds, zero_division=0),
            "best_cv_f1": grid.best_score_,
        }
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        print("Best params:", grid.best_params_)
        print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})
        print(classification_report(ytest, preds, zero_division=0))
        joblib.dump(best, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()
