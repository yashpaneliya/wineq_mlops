import joblib
import yaml
import json
import os
import numpy as np

params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")


class NotInRange(Exception):
    def __init__(self, message="value entered not in range") -> None:
        self.message = message
        super().__init__(self.message)


class NotInColumns(Exception):
    def __init__(self, message="Column not present in data") -> None:
        self.message = message
        super().__init__(self.message)


def read_params(config_path):
    with open(config_path, "r") as yf:
        config = yaml.safe_load(yf)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir = config["webapp_model_dir"]
    model = joblib.load(model_dir)
    pred = model.predict(data).tolist()[0]

    try:
        if 3 <= pred <= 8:
            return pred
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected Result!!"


def get_schema(schema_path):
    with open(schema_path, "r") as jf:
        schema = json.load(jf)
    return schema


def validate(req_dict):

    def _validate_cols(col):
        schema = get_schema(schema_path)
        cols = schema.keys()
        if col not in cols:
            raise NotInColumns

    def _validate_value(col, val):
        schema = get_schema(schema_path)
        if not (schema[col]['min'] <= float(val) <= schema[col]['max']):
            raise NotInRange

    for col, val in req_dict.items():
        _validate_cols(col)
        _validate_value(col, val)

    return True


def form_response(req_dict):
    if validate(req_dict):
        data = req_dict.values()
        data = [list(map(float, data))]
        res = predict(data)
        return res


def api_response(req_dict):
    try:
        if validate(req_dict):
            data = np.array([list(req_dict.values())])
            res = predict(data)
            res = {"response": res}
            return res
    except Exception as e:
        res = {"expected_range": get_schema(schema_path), "response": str(e)}
        return res
