import csv
import numpy as np

class MnistReader:
    def __init__(self, filename):
        self.inputs = []
        self.targets = []

        with open(filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)

            for i in range(100):
                data = next(reader)
                target, data = data[0], np.array(data[1:], dtype=np.float32)
                
                data = data / 255.0
                
                self.inputs.append(data.reshape(1, 784))

                one_hot = np.zeros((1, 10))
                one_hot[0, int(target)] = 1
                self.targets.append(one_hot)

    def get_inputs(self):
        return self.inputs

    def get_targets(self):
        return self.targets