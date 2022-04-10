# MNIST_Model_Visualization

![GitHub Logo](/Example.JPG)

## Summary

Run main.py (need pygame and numpy)

This app allow you to draw a digit and will predict in real time the digit while showing you activations for each nodes.

I have written a neural network from scratch (**neural_network** package), trained and saved a model with mnist dataset from keras (**model_training.py**).

**model_visualization** package manage to draw digit on the left and show activations on the right.

## Improvements:

- NeuralNetwork class use very basic algorithm (do not implements batches) which make training very slow.
- Accuracy is not very good (91% for train and test set) especially for some digit like 0, 9
- Implementing a better way to represent nodes.
- Graph doesn't show weight and bias yet.

