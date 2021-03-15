import argparse
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import Text

from src.utils import load_config


def train(config_path: Text) -> None:
    """Train model
    Args:
       config_path {Text}: path to config
    """

    config = load_config(config_path)
    # Load train set
    train_dataset = pd.read_csv(config.data_split.train_path)

    # Get X and Y
    y = train_dataset.loc[:, 'target'].values.astype('float32')
    X = train_dataset.drop('target', axis=1).values

    # Create an instance of classifier and fit the data.
    clf = LogisticRegression(C=0.00001, solver='lbfgs', multi_class='multinomial', max_iter=100)
    clf.fit(X, y)

    joblib.dump(clf, config.train.model_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
