import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse
from get_data import read_params

import argparse
import joblib
import json

import mlflow


def eval_metrics(y_test, preds):
    rmse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)

    train_path = config["split_data"]["train_dataset"]
    test_path = config["split_data"]["test_dataset"]
    random_state = config["base"]["random_state"]

    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = config["base"]["target_col"]

    traindf = pd.read_csv(train_path)
    testdf = pd.read_csv(test_path)

    X_train = traindf.drop(target, axis=1)
    X_test = testdf.drop(target, axis=1)

    y_train = traindf[target]
    y_test = testdf[target]

    # MLFLOW SETUP
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config['remote_server_uri']

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config['experiment_name'])

    with mlflow.start_run(run_name=mlflow_config['run_name']) as mlops_run:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                           random_state=random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, preds)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        print(tracking_url_type_store)
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, "model", registered_model_name=mlflow_config['registered_model_name'])
        else:
            mlflow.sklearn.log_model(
                model, "model", registered_model_name=mlflow_config['registered_model_name'])
            # mlflow.sklearn.load_model(model, "model") 
        # print("RMSE: ", rmse)
        # print("MAE: ", mae)
        # print("R2-Score: ", r2)

        # scores_file = config["reports"]["scores"]
        # with open(scores_file, "w") as f:
        #     scores = {
        #         "rmse" : rmse,
        #         "mae": mae,
        #         "r2": r2
        #     }
        #     json.dump(scores, f, indent=4)

        # params_file = config["reports"]["params"]
        # with open(params_file, "w") as f:
        #     params = {
        #         "alpha" : alpha,
        #         "l1_ratio": l1_ratio
        #     }
        #     json.dump(params, f, indent=4)

        # os.makedirs(model_dir, exist_ok=True)
        # model_path = os.path.join(model_dir, "model.joblib")

        # joblib.dump(model, model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")

    parsed_args = args.parse_args()

    train_and_evaluate(parsed_args.config)
