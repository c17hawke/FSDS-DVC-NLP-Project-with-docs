import os
import yaml
import logging
import time
import pandas as pd
import json

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content

def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")


def get_df(
    path_to_data: str, 
    sep: str="\t", 
    # column_names: list=["id", "label", "text"],
    encoding='utf-8') -> pd.DataFrame:
    df = pd.read_csv(
        path_to_data, 
        delimiter=sep, 
        encoding=encoding, 
        # header=None, 
        # names=column_names,
    )
    logging.info(f"The input data frame {path_to_data} of size {df.shape} is read.")
    return df