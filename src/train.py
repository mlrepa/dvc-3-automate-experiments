import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib


if __name__ == '__main__':

    # Load train set
    train_dataset = pd.read_csv('data/train.csv')

    # Get X and Y
    y = train_dataset.loc[:, 'target'].values.astype("float32")
    X = train_dataset.drop('target', axis=1).values

    # Create an instance of Logistic Regression Classifier and fit the data.
    clf = LogisticRegression(C=0.01, solver='lbfgs', multi_class='multinomial', max_iter=100)
    # clf = SVC(C=0.1, kernel='linear', gamma='scale', degree=5)
    clf.fit(X, y)

    joblib.dump(clf, 'data/model.joblib')
