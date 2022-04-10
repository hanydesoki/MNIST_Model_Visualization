import numpy as np
import matplotlib.pyplot as plt

import tqdm

import pickle

class NeuralNetwork:

    def __init__(self, hidden_layers: tuple, learning_rate: float = 0.1, n_iter: int = 1000, random_state=None):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def param_initialisation(self, X: np.ndarray, y: np.ndarray):

        dimensions = list(self.hidden_layers)
        dimensions.insert(0, X.shape[0])
        dimensions.append(y.shape[0])

        self.params = None

        params = {}

        self.loss = []
        self.iterations = []

        if self.random_state is not None:
            np.random.seed(self.random_state)

        C = len(dimensions)

        for c in range(1, C):
            params[f"W{c}"] = np.random.randn(dimensions[c], dimensions[c - 1])
            params[f"b{c}"] = np.random.randn(dimensions[c], 1)

        return params

    def forward_propagation(self, X: np.ndarray, params: dict):

        activations = {'A0': X}

        C = len(params) // 2

        for c in range(1, C + 1):
            Z = params[f'W{c}'].dot(activations[f'A{c - 1}']) + params[f'b{c}']
            activations[f'A{c}'] = self.sigmoid(Z)

        return activations

    def back_propagation(self, y: np.ndarray, activations: dict, params: dict):

        m = y.shape[0]
        C = len(params) // 2

        dZ = activations[f'A{C}'] - y

        gradients = {}

        for c in reversed(range(1, C + 1)):
            gradients[f'dW{c}'] = 1 / m * np.dot(dZ, activations[f'A{c - 1}'].T)
            gradients[f'db{c}'] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(params[f'W{c}'].T, dZ) * activations[f'A{c - 1}'] * (1 - activations[f'A{c - 1}'])

        return gradients

    def gradient_descent(self, gradients: dict, params: dict):

        C = len(params) // 2

        for c in range(1, C + 1):
            params[f'W{c}'] -= self.learning_rate * gradients[f'dW{c}']
            params[f'b{c}'] -= self.learning_rate * gradients[f'db{c}']

        return params

    def fit(self, X: np.ndarray, y: np.ndarray):

        params = self.param_initialisation(X, y)
        C = len(params) // 2

        for i in tqdm.tqdm(range(self.n_iter)):

            activations = self.forward_propagation(X, params)
            gradients = self.back_propagation(y, activations, params)
            params = self.gradient_descent(gradients, params)

            if i % 10 == 0:
                self.loss.append(self.log_loss(activations[f'A{C}'], y))
                self.iterations.append(i)

        self.params = params

        return self


    def predict(self, X: np.ndarray):

        activations = self.forward_propagation(X, self.params)
        C = len(self.params) // 2
        Af = activations[f'A{C}']
        return Af

    def plot_loss(self, figsize: tuple = (8, 6)):
        if len(self.iterations) < 2:
            raise ValueError('Not enough values to plot')
        plt.figure(figsize=figsize)
        plt.plot(self.iterations, self.loss)
        plt.xlabel('Iterations')
        plt.ylabel('Log loss')
        plt.show()


    @staticmethod
    def log_loss(A, y):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    @staticmethod
    def sigmoid(Z: np.ndarray):
        return 1 / (1 + np.exp(-Z))

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_layers={self.hidden_layers}, learning_rate={self.learning_rate}, n_iter={self.n_iter}, random_state={self.random_state})"