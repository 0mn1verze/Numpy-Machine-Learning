import numpy as np
from nn.layers import Layer

class Dropout(Layer):
    def __init__(self, p: float):
        """Dropout layer

        Args:
            p (float): Dropout probability
        """
        self.p = np.clip(np.array([p]), 1e-6, 1 - 1e-6)[0]

    def setup(self, in_shape, optimiser):
        self.in_shape = in_shape
        self.out_shape = in_shape
        pass

    def forward(self, X):
        self.mask = (np.random.rand(*X.shape) < self.p) / self.p
        Z = X * self.mask
        return Z
    
    def backward(self, dZ):
        dX = dZ * self.mask
        return dX
    
    def params(self):
        return 0, 0