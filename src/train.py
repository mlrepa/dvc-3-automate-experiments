import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib


def train(train_dataset, model_path):

    # Load train set
    train_dataset = pd.read_csv(train_dataset)

    # Get X and Y
    y = train_dataset.loc[:, 'target'].values.astype('float32')
    X = train_dataset.drop('target', axis=1).values

    # Create an instance of classifier and fit the data.
    clf = LogisticRegression(C=0.00001, solver='lbfgs', multi_class='multinomial', max_iter=100)
    clf.fit(X, y)

    joblib.dump(clf, model_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--train-dataset', dest='train_dataset', required=True)
    args_parser.add_argument('--model', dest='model', required=True)
    args = args_parser.parse_args()

    train(args.train_dataset, args.model)