from abc import ABC, abstractmethod
import numpy.typing as npt
from typing import Any
from nn.optimisers import Optimiser

class Layer(ABC):

    @abstractmethod
    def forward(self, X: npt.NDArray[Any]) -> npt.NDArray:
        """Single Layer Forward Propagation

        Args:
            X (npt.NDArray[Any]): Input of the layer

        Returns:
            npt.NDArray[Any]: Output of the layer
        """
        pass

    @abstractmethod
    def backward(self, dZ: npt.NDArray) -> npt.NDArray:
        """Single Layer Backward Propagation

        Args:
            dZ (npt.NDArray): Gradient from the next layer
        
        Returns:
            npt.NDArray: Gradient for the previous layer
        """
        pass

    @abstractmethod
    def setup(self, in_shape: tuple[int] | int, optimiser: Optimiser) -> None:
        """Set up layer

        Args:
            in_shape (tuple[int] | int): Input shape
            optimiser (Optimiser): Optimiser
        """
        pass

    @abstractmethod
    def params(self) -> tuple[int]:
        """Get number of params of the layer

        Returns:
            tuple[int]: number of trainable and non-trainable params
        """
        pass
