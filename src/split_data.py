import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from get_data import read_params

def split_and_save(config_path):
    config = read_params(config_path)
    train_path = config["split_data"]["train_dataset"]
    test_path = config["split_data"]["test_dataset"]

    raw_path = config["load_data"]["raw_dataset"]
    split_ratio = config["split_data"]["test_size"]

    random_state = config["base"]["random_state"]

    df = pd.read_csv(raw_path, sep=',', encoding='utf-8')

    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_path, sep=',', index=False, encoding='utf-8')
    test.to_csv(test_path, sep=',', index=False, encoding='utf-8')

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    
    parsed_args = args.parse_args()

    split_and_save(parsed_args.config)  