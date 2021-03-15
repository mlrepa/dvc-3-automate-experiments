import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Text

from src.utils import load_config


def split_train_test(config_path: Text) -> None:
    """Split dataset into train and test
    Args:
       config_path {Text}: path to config
    """

    config = load_config(config_path)
    dataset = pd.read_csv(config.featurize.features_path)

    # Split in train/test
    test_size = config.data_split.test_size
    df_train, df_test = train_test_split(dataset, test_size=test_size, random_state=42)

    df_train.to_csv(config.data_split.train_path, index=False)
    df_test.to_csv(config.data_split.test_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    split_train_test(config_path=args.config)
