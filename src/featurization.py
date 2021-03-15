import argparse
import pandas as pd
from typing import Text

from src.utils import load_config


def get_features(dataset):

    features = dataset.copy()

    return features


def featurize(config_path: Text) -> None:
    """Create features
    Args:
        config_path {Text}: path to config
    """

    config = load_config(config_path)

    dataset = pd.read_csv(config.data_load.raw_data_path)
    features = get_features(dataset)
    features.to_csv(config.featurize.features_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)
