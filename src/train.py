import argparse
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import Text
import yaml


def train(config_path: Text) -> None:
    """Train model
    Args:
       config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    train_dataset_path = config['data_split']['train_path']
    model_path = config['train']['model_path']
    # Load train set
    train_dataset = pd.read_csv(train_dataset_path)

    # Get X and Y
    y = train_dataset.loc[:, 'target'].values.astype('float32')
    X = train_dataset.drop('target', axis=1).values

    # Create an instance of classifier and fit the data.
    clf = LogisticRegression(C=0.00001, solver='lbfgs', multi_class='multinomial', max_iter=100)
    clf.fit(X, y)

    joblib.dump(clf, model_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)