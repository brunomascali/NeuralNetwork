import numpy as np
from activation.activation import *

class InputLayer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.output = np.zeros((input_size, 1))

    def forward(self, input):
        self.input = input
        self.output = input.reshape(1, -1)
        return self.output

class DenseLayer:
    def __init__(self, input_size, output_size, activation='sig'):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

        self.input = None
        self.output = None
        self.delta = None

        match activation:
            case 'sig':
                self.activation_function = Sigmoid()
            case 'relu':
                self.activation_function = ReLU()
            case 'softmax':
                self.activation_function = Softmax()
            case _:
                raise ValueError(f"Invalid activation function: {activation}")


    def forward(self, input):
        self.input = input
        
        self.input = input @ self.weights + self.bias
        self.output = self.activation_function(self.input)
        return self.output