import pandas as pd
from sklearn.model_selection import train_test_split


def featurize(data):
    X = pd.DataFrame(data.data)
    X.columns = data.feature_names
    y = pd.DataFrame(data.target)
    y.columns = ['target']
    return X, y


def transform_data(data):
    X, y = featurize(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
