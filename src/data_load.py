import argparse
import json
from sklearn.datasets import load_iris
from typing import Text
import yaml


def data_load(config_path: Text) -> None:
    """Load raw data

    Args:
        config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    raw_data_path = config['data_load']['raw_data_path']
    classes_names_path = config['data_load']['classes_names_path']

    data = load_iris(as_frame=True)
    classes_names = data.target_names.tolist()

    dataset = data.frame
    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]
    dataset.to_csv(raw_data_path, index=False)

    with open(classes_names_path, 'w') as classes_names_file:
        json.dump(obj={'classes_names': classes_names}, fp=classes_names_file)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
