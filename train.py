import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from data_prep import featurize


def train(max_depth: int) -> None:
    # Load data
    data = load_iris()

    # Get features
    X, y = featurize(data)

    # Train model
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X, y)

    # Save model to disk
    with open("tree_classifier.pkl", "wb") as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    train(max_depth=2)
