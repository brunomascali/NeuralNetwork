# Simple Neural Network Implementation

A basic neural network implementation built from scratch using only NumPy.

## Features

- Customizable neural network architecture
- Multiple activation functions (Sigmoid, ReLU)
- Various loss functions (MSE, Binary Cross Entropy, Categorical Cross Entropy)
- Dense layer implementation
- MNIST digit classification example

## Project Structure

```
├── neural_network.py     # Main neural network implementation
├── layer
│   └── dense_layer.py    # Dense layer implementation
├── activation/
│   └── activation.py     # Activation functions
├── cost/
│   └── cost.py           # Cost functions
├── main.py               # MNIST training example
├── mnist_reader.py       # MNIST data loader
```


## Usage

Basic example for training a simple neural network:

```python
from neural_network import NeuralNetwork
from layer.dense_layer import DenseLayer, InputLayer
from activation import Sigmoid

# Create network
nn = NeuralNetwork(layers=[
    InputLayer(input_size=2),
    DenseLayer(input_size=2, output_size=4, activation=Sigmoid),
    DenseLayer(input_size=4, output_size=1, activation=Sigmoid)
], cost='bce')

# Train on XOR data
xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_targets = [[0], [1], [1], [0]]
nn.train(xor_inputs, xor_targets, epochs=1000, learning_rate=0.1)
```

## MNIST Example

The project includes a complete MNIST digit classification example:

## Future Goals

implement Convolutional Neural Networks (CNNs) to extend this simple implementation to handle more complex computer vision tasks.

## Dependencies

- NumPy >= 1.21.0