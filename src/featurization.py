import pandas as pd


def get_features(dataset):

    features = dataset.copy()

    # uncomment for step 5.2  Add features
    # features['sepal_length_to_sepal_width'] = features['sepal_length'] / features['sepal_width']
    # features['petal_length_to_petal_width'] = features['petal_length'] / features['petal_width']

    return features


if __name__ == '__main__':

    dataset = pd.read_csv('data/iris.csv')

    features  = get_features(dataset)
    features.to_csv('data/iris_featurized.csv', index=False)
