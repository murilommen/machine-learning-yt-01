import pickle
import numpy as np


def load_model():
    with open("tree_classifier.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def load_data() -> np.array:
    return np.random.rand(10, 4) * 4


def predict(data, model) -> np.array:
    predictions = model.predict(data)
    return predictions


if __name__ == "__main__":
    loaded_model = load_model()
    new_data = load_data()

    print(predict(data=new_data, model=loaded_model).tolist())
