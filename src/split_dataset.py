import argparse
import pandas as pd

from sklearn.model_selection import train_test_split


def split_train_test(featurized_dataset_path, train_dataset_path,
                     test_dataset_path, test_size):

    dataset = pd.read_csv(featurized_dataset_path)

    # transform targets (species) to numerics
    # dataset.loc[dataset.species=='setosa', 'species'] = 0
    # dataset.loc[dataset.species=='versicolor', 'species'] = 1
    # dataset.loc[dataset.species=='virginica', 'species'] = 2

    # Split in train/test

    df_train, df_test = train_test_split(dataset, test_size=test_size, random_state=42)

    df_train.to_csv(train_dataset_path, index=False)
    df_test.to_csv(test_dataset_path, index=False)


if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--featurized-dataset', dest='featurized_dataset', required=True)
    arg_parser.add_argument('--train-dataset', dest='train_dataset', required=True)
    arg_parser.add_argument('--test-dataset', dest='test_dataset', required=True)
    arg_parser.add_argument('--test-size', dest='test_size', type=float)

    args = arg_parser.parse_args()

    split_train_test(
        args.featurized_dataset,
        args.train_dataset,
        args.test_dataset,
        args.test_size
    )
