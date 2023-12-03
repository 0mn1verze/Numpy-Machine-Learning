import numpy as np
from abc import ABC, abstractmethod
import numpy.typing as npt
from typing import Any

class Decay:
    @abstractmethod
    def update():
        pass

class Constant(Decay):
    def update(self, t: int, lr_0: float) -> float:
        """Update

        Args:
            t (int): Iteration
            lr_0 (float): Initial learning rate

        Returns:
            float: Updated learning rate
        """
        return lr_0

class TimeDecay(Decay):
    def update(self, t:int, lr_0: float, k: float) -> float:
        """Update

        Args:
            t (int): Iteration
            lr_0 (float): Initial learning rate
            k (float): Decay rate

        Returns:
            float: Updated learning rate
        """
        return lr_0/(1+k*t)

class StepDecay(Decay):
    def update(self, t: int, lr_0: float, F: float, D: float) -> float:
        """Update

        Args:
            t (int): Iteration
            lr_0 (float): Initial learning rate
            F (float): Drop factor
            D (float): Iterations per drop

        Returns:
            float: Updated learning rate
        """

        mult = F**np.floor((1+t/D))
        lr = lr_0 * mult

        return lr

class ExpDecay(Decay):
    def update(self, t: int, lr_0: float, k: float) -> float:
        """Update

        Args:
            t (int): Iteration
            lr_0 (float): Initial learning rate
            k (float): Exponential decay rate

        Returns:
            float: Updated learning rate
        """
        return lr_0 * np.exp(-k**t)
