import os
import yaml
import pandas as pd
import argparse
from get_data import read_params, get_data

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)

    new_cols = [col.replace(' ', '_') for col in df.columns]
    print(new_cols)

    raw_path = config["load_data"]["raw_dataset"]
    df.to_csv(raw_path, index=False, header=new_cols, sep=',', encoding='utf-8')

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    
    parsed_args = args.parse_args()

    load_and_save(parsed_args.config)  