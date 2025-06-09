import numpy as np

class Sigmoid:
    def __call__(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        x = np.clip(x, -500, 500)
        sig = self(x)
        return sig * (1 - sig)

class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class Softmax:
    def __call__(self, x):
        x = np.clip(x, -500, 500)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    def derivative(self, x):
        return self(x) * (1 - self(x))
    
