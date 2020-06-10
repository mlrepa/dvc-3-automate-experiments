import argparse
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import joblib


def evaluate(raw_dataset_path, test_dataset_path, model_path,  eval_report_path):

    classes = pd.read_csv(raw_dataset_path)['target'].unique().tolist()

    test_dataset = pd.read_csv(test_dataset_path)
    y = test_dataset.loc[:, 'target'].values.astype('float32')
    X = test_dataset.drop('target', axis=1).values

    clf = joblib.load(model_path)

    prediction = clf.predict(X)
    cm = confusion_matrix(prediction, y)
    f1 = f1_score(y_true=y, y_pred=prediction, average='macro')

    json.dump(
        obj={
            'f1_score': f1,
            'confusion_matrix': {
                'classes': classes,
                'matrix': cm.tolist()
            }
        },
        fp=open(eval_report_path, 'w')
    )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--raw-dataset', dest='raw_dataset', required=True)
    args_parser.add_argument('--test-dataset', dest='test_dataset', required=True)
    args_parser.add_argument('--model', dest='model', required=True)
    args_parser.add_argument('--eval-report', dest='eval_report', required=True)
    args = args_parser.parse_args()

    evaluate(
        args.raw_dataset,
        args.test_dataset,
        args.model,
        args.eval_report
    )

