from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from data_prep import transform_data


def evaluate(max_depth: int) -> None:
    # Load data
    data = load_iris()

    # Get features and split into train & test
    X_train, X_test, y_train, y_test = transform_data(data)

    # Train model
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    # Evaluate model
    y_preds = model.predict(X_test)
    acc_score = accuracy_score(y_true=y_test, y_pred=y_preds)
    f_score = f1_score(y_true=y_test, y_pred=y_preds, average='weighted')

    # Print metrics
    print("accuracy score is: " + str(acc_score))
    print("F1 score is: " + str(f_score))


if __name__ == "__main__":
    evaluate(max_depth=1)
