from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Any
from nn.optimisers import Optimiser

class Activation(ABC):
    @abstractmethod
    def forward(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Activation Layer Forward Propagation

        Args:
            X (npt.NDArray[Any]): Input

        Returns:
            npt.NDArray[Any]: Output
        """
        pass

    @abstractmethod
    def backward(self, dZ: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Activation Layer Backward Propagation

        Args:
            dZ (npt.NDArray[Any]): Gradient from next layer

        Returns:
            npt.NDArray[Any]: Gradient for previous layer
        """
        pass
    
    def setup(self, in_shape: tuple[int] | int, optimser: Optimiser):
        """Set up layer

        Args:
            in_shape (tuple[int] | int): Input shape
        """
        self.in_shape = in_shape
        self.out_shape = in_shape

class Linear(Activation):
    def forward(self, X):
        self.X = X
        return X
    
    def backward(self, dZ):
        return dZ * np.ones(self.X.shape)
    
class Sigmoid(Activation):
    def forward(self, X):
        self.Z = 1 / (1 + np.exp(-X))
        return self.Z

    def backward(self, dZ):
        return dZ * self.Z * (1 - self.Z)
    
class Tanh(Activation):
    def forward(self, X):
        self.Z = np.tanh(X)
        return self.Z

    def backward(self, dZ):
        return dZ * (1 - self.Z ** 2)
    
class ReLU(Activation):
    def forward(self, X):
        self.X = X
        return self.X * (self.X > 0)

    def backward(self, dZ):
        return dZ * (np.ones(self.X.shape) * (self.X > 0))

class ParaReLU(Activation):
    def forward(self, X, alpha=0.01):
        self.X = X
        return np.where(self.X > 0, self.X, alpha*self.X)

    def backward(self, dZ, alpha=0.01):
        return dZ * np.where(self.X > 0, 1, alpha)

class SoftMax(Activation):
    def forward(self, X):
        self.X = X
        z = np.array(self.X) - np.max(self.X, axis=-1, keepdims=True)
        num = np.exp(z)
        den = np.sum(num, axis=-1, keepdims=True)
        self.Z = num / den
        return self.Z
    
    def backward(self, dZ):
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(1, -1)
        _, d = self.X.shape

        t1 = np.einsum('ij,ik->ijk', self.Z, self.Z)
        t2 = np.einsum('ij,jk->ijk', self.Z, np.eye(d, d))
        
        return np.einsum('ij,ijk->ik', dZ, t2-t1)