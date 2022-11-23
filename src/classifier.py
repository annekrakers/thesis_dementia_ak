from typing import Union

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

METRICS = [f1_score, accuracy_score]


def split_train_test(
    X: pd.DataFrame, y: pd.DataFrame, test_size: int = 0.2, random_state: int = None
) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset in a train and test dataset.
    """
    return train_test_split(X, y, random_state=random_state, test_size=test_size)


def fit_classifier(clf: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """
    Fit the classifier on the given input data.
    """
    clf.fit(X_train, y_train)
    return clf


def predict(trained_clf: Pipeline, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the values for the input.
    """
    return trained_clf.predict(X_test)


def eval_clf(y_test: pd.DataFrame, y_pred: pd.DataFrame, print_results: bool = False) -> dict:
    """
    Evaluate the score for the classifier.
    """
    results = {}
    for metric in METRICS:
        results[metric.__name__] = metric(y_test, y_pred)

    if print_results:
        print(results)
    return results
