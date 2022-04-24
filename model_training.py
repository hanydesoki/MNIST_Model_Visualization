from neural_network import NeuralNetwork, load_model

from predict import predict, mnist_score

from keras.datasets import mnist
from keras.utils import np_utils


def main() -> None:

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 28 * 28)
    X_train = X_train.astype('float32')
    X_train /= 255

    y_train = np_utils.to_categorical(y_train)

    X_test = X_test.reshape(X_test.shape[0], 28 * 28)
    X_test = X_test.astype('float32')
    X_test /= 255
    y_test = np_utils.to_categorical(y_test)

    mnist_model = NeuralNetwork(hidden_layers=(128,), learning_rate=0.0025, n_iter=500, random_state=0)
    print(mnist_model)
    mnist_model.fit(X_train.T, y_train.T)

    mnist_model.save('mnist_model.pkl')

    mnist_model = load_model('mnist_model.pkl')

    y_pred_train = predict(mnist_model, X_train)
    y_pred_test = predict(mnist_model, X_test)

    print(f'Train accuracy:', mnist_score(y_train, y_pred_train))
    print(f'Test accuracy:', mnist_score(y_test, y_pred_test))

    mnist_model.plot_loss()


if __name__ == '__main__':
    main()







