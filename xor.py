from layer.dense_layer import DenseLayer, InputLayer
from neural_network import NeuralNetwork
import numpy as np

if __name__ == "__main__":
    xor_gate = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    xor_gate_output = np.array([[0], [1], [1], [0]], dtype=np.float32)

    print("Training XOR gate...")

    nn = NeuralNetwork(layers=[
        InputLayer(input_size=2), 
        DenseLayer(input_size=2, output_size=2),
        DenseLayer(input_size=2, output_size=1),
    ], cost='bce')
    
    nn.train(xor_gate, xor_gate_output, epochs=10000, learning_rate=0.05)

    for comb in xor_gate:
        print(nn.forward(comb.reshape(2, 1)))