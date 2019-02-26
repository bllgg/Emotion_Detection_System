import math
import numpy as np

class Connection:
    def __init__(self,connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0

