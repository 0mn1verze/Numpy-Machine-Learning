import numpy as np
import numpy.typing as npt
from typing import Any
from nn.layers import Layer
from nn.initialisers import Weight, He

class Dense(Layer):
    def __init__(self, out_shape: int, use_bias: bool = True, seed: int = 42, initialiser: Weight = He(), regulariser: tuple[str, float] = ('L2', 0)):
        self.out_shape = out_shape
        self.initialiser = initialiser
        self.regulariser = regulariser
        self.use_bias = use_bias
        self.seed = seed
        self.W: npt.NDArray[Any] # Weights
        self.b: npt.NDArray[Any] # Bias
        self.X: npt.NDArray[Any] # Input
        self.Y: npt.NDArray[Any] # Output

    def setup(self, in_shape, optimiser):
        """Set up layer parameters

        Args:
            in_shape (int): Input shape
            optimiser (Class): Optimiser for updating parameters
        """
        self.in_shape = in_shape

        self.W_shape = (self.in_shape, self.out_shape)
        self.b_shape = (1, self.out_shape)

        self.optimiser = optimiser
        self.W = self.initialiser.initialise((self.in_shape, self.out_shape), seed=self.seed)
        self.b = np.zeros(self.b_shape)

        self.optimiser = optimiser(self.W_shape, self.b_shape)

    def forward(self, X):
        self.X = X # Caching input for backward prop

        return X @ self.W + self.b
    
    def backward(self, dZ):
        dX = dZ @ self.W.T # Input gradient
        self.db = np.sum(dZ, axis=0) # Bias gradient
        self.dW = self.X.T @ dZ # Weight gradient

        return dX
    
    def update(self, lr: float, m: int, k: int):
        """Update Dense Layer Parameters

        Args:
            lr (float): Learning rate
            m (int): Batch size
            k (int): Iteration
        """

        dW, db = self.optimiser.update(self.dW, self.db, k)

        if self.regulariser[0].lower() == 'l2':
            dW += self.regulariser[1] * self.W
        elif self.regulariser[0].lower() == 'l1':
            dW += self.regulariser[1] * np.sign(self.W)

        self.W -= dW * (lr/m)
        if self.use_bias:
            self.b -= db * (lr/m)

    def params(self):
        no_of_params = self.in_shape * self.out_shape
        if self.use_bias:
            no_of_params += self.out_shape
        return no_of_params, 0