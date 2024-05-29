import yaml
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os

def read_params(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def log_production_model(config_path):
    config = read_params(config_path)

    mlflow_config = config['mlflow_config']

    remote_server_uri = mlflow_config['remote_server_uri']
    model_name = mlflow_config['registered_model_name']

    mlflow.set_tracking_uri(remote_server_uri)
    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]

    runs = mlflow.search_runs(experiment_ids=["1"])
    print(runs.columns)

    lowest_mae_val = runs['metrics.MAE'].sort_values(ascending=True)[0]
    lowest_run_id = runs[runs['metrics.MAE']==lowest_mae_val]["run_id"][0]

    print(lowest_run_id)
    
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        print(mv['run_id'])
        if mv["run_id"] == lowest_run_id:
            current_version = mv['version']
            logged_model = mv['source']
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production",
            )
        else:
            current_version = mv['version']
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging",
            )
    
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_path = config['webapp_model_dir']

    joblib.dump(loaded_model, model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")

    parsed_args = args.parse_args()

    log_production_model(parsed_args.config)