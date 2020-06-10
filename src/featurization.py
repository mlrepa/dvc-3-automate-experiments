import argparse
import pandas as pd


def get_features(dataset):

    features = dataset.copy()

    return features


def featurize(raw_dataset_path, featurized_dataset_path):

    dataset = pd.read_csv(raw_dataset_path)
    features = get_features(dataset)
    features.to_csv(featurized_dataset_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--raw-dataset', dest='raw_dataset', required=True)
    args_parser.add_argument('--featurized-dataset', dest='featurized_dataset', required=True)
    args = args_parser.parse_args()

    featurize(args.raw_dataset, args.featurized_dataset)
