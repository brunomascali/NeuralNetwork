import numpy as np
import pickle
from cost.cost import BCE, MSE, CCE

class NeuralNetwork:
    def __init__(self, layers, cost='bce'):
        self.layers = layers
        match cost:
            case 'bce':
                self.cost_function = BCE()
            case 'mse':
                self.cost_function = MSE()
            case 'cce':
                self.cost_function = CCE()
            case _:
                raise ValueError(f"Invalid cost function: {cost}")

    def forward(self, input):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.forward(input)
            else:
                layer.forward(self.layers[i - 1].output)
            
        return self.layers[-1].output

    def backpropagation(self, output, expected_output):
        self.layers[-1].delta = self.cost_function.derivative(expected_output, output)

        for i in range(len(self.layers) - 2, 0, -1):
            next_layer = self.layers[i + 1]
            self.layers[i].delta = (
                next_layer.delta @ next_layer.weights.T * 
                self.layers[i].activation_function.derivative(self.layers[i].input)
            )

    def gradient_descent(self, learning_rate):
        for i in range(1, len(self.layers)):
            current_layer = self.layers[i]
            previous_layer = self.layers[i - 1]

            current_layer.weights -= learning_rate * (previous_layer.output.T @ current_layer.delta)
            current_layer.bias -= learning_rate * np.sum(current_layer.delta, axis=0, keepdims=True)

    def train(self, input, expected_output, epochs=5000, learning_rate=0.1, batch_size=32):
        num_samples = len(input)
        
        for i in range(epochs):
            indices = np.random.permutation(num_samples)
            
            for j in range(0, num_samples, batch_size):
                batch_indices = indices[j:j + batch_size]
                batch_inputs = [input[idx] for idx in batch_indices]
                batch_targets = [expected_output[idx] for idx in batch_indices]
                
                batch_results = []
                for x in batch_inputs:
                    result = self.forward(x)
                    batch_results.append(result)
                
                for y, result in zip(batch_targets, batch_results):
                    self.backpropagation(result, y)
                
                self.gradient_descent(learning_rate)

            # Calculate and print error every 50 epochs
            if i % 50 == 0:
                total_error = 0
                for x, y in zip(input, expected_output):
                    result = self.forward(x)
                    total_error += self.cost_function.cost(y, result)
                avg_error = total_error / num_samples
                print(f"Epoch {i}, Average Error: {avg_error:.6f}")

    def serialize(self, filename):
        state = {
            'layers': self.layers,
            'cost_function': self.cost_function
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def deserialize(self, filename):
        with open(filename, 'rb') as f: 
            state = pickle.load(f)
            self.layers = state['layers']
            self.cost_function = state['cost_function']

    def load(self, filename):
        self.deserialize(filename)
        return self