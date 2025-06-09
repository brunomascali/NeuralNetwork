from sympy import O
from layer.dense_layer import DenseLayer, InputLayer
from neural_network import NeuralNetwork
from cost.cost import BCE

import numpy as np

if __name__ == "__main__":
    and_gate = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_gate_output = np.array([[0], [0], [0], [1]])

    nn = NeuralNetwork(layers=
                       [InputLayer(input_size=2), 
                        DenseLayer(input_size=2, output_size=1)
                        ], 
                        cost=BCE)
    
    nn.train(and_gate, and_gate_output)


    for comb in and_gate:
        print(nn.forward(comb.reshape(2, 1)))