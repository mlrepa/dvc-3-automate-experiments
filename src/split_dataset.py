import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Text
import yaml


def split_train_test(config_path: Text) -> None:
    """Split dataset into train and test
    Args:
       config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    featurized_dataset_path = config['featurize']['features_path']
    train_dataset_path = config['data_split']['train_path']
    test_dataset_path = config['data_split']['test_path']
    test_size = config['data_split']['test_size']

    dataset = pd.read_csv(featurized_dataset_path)

    # Split in train/test

    df_train, df_test = train_test_split(dataset, test_size=test_size, random_state=42)

    df_train.to_csv(train_dataset_path, index=False)
    df_test.to_csv(test_dataset_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    split_train_test(config_path=args.config)
