import argparse
import joblib
import json
import pandas as pd
from sklearn.metrics import f1_score
from typing import Text
import yaml


def evaluate(config_path: Text) -> None:
    """Evaluate model
    Args:
       config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    classes_names_path = config['data_load']['classes_names_path']
    test_dataset_path = config['data_split']['test_path']
    model_path = config['train']['model_path']
    metrics_path = config['evaluate']['metrics_file']
    confusion_matrix_path = config['evaluate']['confusion_matrix']

    classes = json.load(open(classes_names_path))['classes_names']

    test_dataset = pd.read_csv(test_dataset_path)
    y = test_dataset.loc[:, 'target'].values.astype('float32')
    X = test_dataset.drop('target', axis=1).values

    clf = joblib.load(model_path)

    prediction = clf.predict(X)
    f1 = f1_score(y_true=y, y_pred=prediction, average='macro')

    json.dump(
        obj={'f1_score': f1},
        fp=open(metrics_path, 'w')
    )

    mapping = {i: cls_name for i, cls_name in enumerate(classes)}
    cmdf = pd.DataFrame(
        {'actual': y, 'predicted': prediction}
    ).apply(lambda series: series.map(mapping))
    cmdf.to_csv(confusion_matrix_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config)

