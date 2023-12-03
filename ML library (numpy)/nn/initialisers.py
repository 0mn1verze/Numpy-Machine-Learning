import numpy as np
from abc import ABC, abstractmethod
import numpy.typing as npt
from typing import Any

class Weight(ABC):
    @abstractmethod
    def initialise(self, shape: tuple[int], seed: int=None) -> npt.NDArray[Any]:
        """Dense Layer Weight Initialisation

        Args:
            shape (tuple[int]): Shape of the weights/kernels

        Returns:
            npt.NDArray: Randomised Weights
        """
        pass

class Zeros(Weight):
    """Zero initialisation"""
    def initialise(self, shape, seed):
        if seed:
            np.random.seed(seed)
        return np.zeros(shape)
    
class Ones(Weight):
    """One initialisation"""
    def initialise(self, shape, seed):
        if seed:
            np.random.seed(seed)
        return np.ones(shape)
    
class RandNorm(Weight):
    """Random Normal initialisation"""
    def initialise(self, shape, seed):
        if seed:
            np.random.seed(seed)
        return np.random.normal(size=shape)
    
class RandUniform(Weight):
    """Random Uniform initialisation"""
    def initialise(self, shape, seed):
        if seed:
            np.random.seed(seed)
        return np.random.uniform(size=shape)

class He(Weight):
    """He initialisation for ReLU activation"""
    def initialise(self, shape, seed):
        if seed:
            np.random.seed(seed)
        try:
            F, Kc, Kh, Kw = shape
        except:
            Kh, Kw = shape
        return np.random.randn(*shape) * np.sqrt(2 / Kh)

class Xavier(Weight):
    """Xavier initialisation for Tanh activation"""
    def initialise(self, shape, seed):
        if seed:
            np.random.seed(seed)
        try:
            F, Kc, Kh, Kw = shape
        except:
            Kh, Kw = shape
        return np.random.randn(*shape) * np.sqrt(1 / Kh)
    
class Glorot(Weight):
    """Glorot initialisation for Tanh activation"""
    def initialise(self, shape, seed):
        if seed:
            np.random.seed(seed)
        try:
            F, Kc, Kh, Kw = shape
        except:
            Kh, Kw = shape
        return np.random.randn(*shape) * np.sqrt(2 / (Kh + Kw))