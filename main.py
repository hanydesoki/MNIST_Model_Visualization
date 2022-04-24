from model_visualization import App
from neural_network import load_model


def main() -> None:

    model_path = 'mnist_model.pkl'

    mnist_model = load_model(model_path)

    app = App(model=mnist_model)

    app.run()


if __name__ == '__main__':
    main()
