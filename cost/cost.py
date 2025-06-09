import numpy as np

class MSE:
    def cost(self, target, prediction):
        return np.mean((target - prediction) ** 2)

    def derivative(self, target, prediction):
        return 2 * (prediction - target) / target.size
    
class BCE:
    def cost(self, target, prediction):
        eps = 1e-15
        prediction = np.clip(prediction, eps, 1 - eps)
        return -np.sum(target * np.log(prediction) + (1 - target) * np.log(1 - prediction)) / target.size
    
    def derivative(self, target, prediction):
        return prediction - target

class CCE:
    def cost(self, target, prediction):
        eps = 1e-15
        prediction = np.clip(prediction, eps, 1 - eps)
        return -np.sum(target * np.log(prediction)) / target.size
    
    def derivative(self, target, prediction):
        return prediction - target