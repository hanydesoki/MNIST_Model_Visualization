from neural_network import NeuralNetwork

import numpy as np

def predict(model: NeuralNetwork, X: np.ndarray):
    return model.predict(X.T).T

def mnist_score(y_true: np.ndarray, y_pred: np.ndarray):
    return (np.array([np.argmax(y) for y in y_true]) == np.array([np.argmax(y) for y in y_pred])).mean()