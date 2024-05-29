import pytest
import yaml
import os
import json

@pytest.fixture
def config(config_path="params.yaml"):
    with open(config_path, "r") as yf:
        config = yaml.safe_load(yf)
    return config

@pytest.fixture
def schema(schema_path="schema_in.json"):
    with open(schema_path, "r") as jf:
        schema = json.load(jf)
    return schema