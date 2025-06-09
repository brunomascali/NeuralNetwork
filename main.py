import numpy as np
from layer.dense_layer import DenseLayer, InputLayer
from activation.activation import Sigmoid
from neural_network import NeuralNetwork
from mnist_reader import MnistReader

# https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

if __name__ == "__main__":
    mnist_reader = MnistReader('mnist_train.csv')
    inputs = mnist_reader.get_inputs()
    targets = mnist_reader.get_targets()

    print(f"Loaded {len(inputs)} samples")
    print(f"Input shape: {inputs[0].shape}")
    print(f"Target shape: {targets[0].shape}")

    nn = NeuralNetwork(layers=[
        InputLayer(input_size=784), 
        DenseLayer(input_size=784, output_size=200, activation='relu'),
        DenseLayer(input_size=200, output_size=100, activation='sig'),
        DenseLayer(input_size=100, output_size=10, activation='sig'),
    ], cost='cce')
    
    nn.train(inputs, targets, epochs=15000, learning_rate=0.01)

    mnist_reader = MnistReader('mnist_test.csv')
    inputs = mnist_reader.get_inputs()
    targets = mnist_reader.get_targets()

    correct = 0
    for i in range(len(inputs)):
        prediction = nn.forward(inputs[i])
        predicted_digit = np.argmax(prediction[0])
        actual_digit = np.argmax(targets[i][0])
        
        if predicted_digit == actual_digit:
            correct += 1

    print(f"Accuracy: {correct / len(inputs) * 100}%")