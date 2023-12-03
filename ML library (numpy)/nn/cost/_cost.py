from abc import ABC, abstractmethod
import numpy.typing as npt
from typing import Any

class Cost(ABC):
    funcs = {}
    @abstractmethod
    def cost(self, A: npt.NDArray[Any], Y: npt.NDArray[Any]) -> float:
        """Cost function

        Args:
            A (npt.NDArray[Any]): Prediction from model
            Y (npt.NDArray[Any]): Answer from dataset

        Returns:
            float: Cost
        """
        pass

    @abstractmethod
    def backward(self, A: npt.NDArray[Any], Y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Backward propagation

        Args:
            A (npt.NDArray[Any]): Prediction from model
            Y (npt.NDArray[Any]): Answer from dataset

        Returns:
            npt.NDArray[Any]: Gradient for the previous layer
        """
        pass
    