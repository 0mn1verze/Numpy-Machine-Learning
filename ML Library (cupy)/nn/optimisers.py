from abc import ABC, abstractmethod
import cupy as cp
import numpy.typing as npt
from typing import Any

class Optimiser(ABC):
    def __init__(self, W_shape: tuple[int, int], b_shape: tuple[int, int], m1: float=0.9, m2: float=0.999, ep: float=1e-9):
        """Optimiser parameters

        Args:
            W_shape (tuple[int, int]): Shape of the weight
            b_shape (tuple[int, int]): Shape of the bias
            m1 (float, optional): Momentum that accelerates gradient descent. Defaults to 0.9.
            m2 (float, optional): Momentum (Only used in Adam). Defaults to 0.999.
            ep (_type_, optional): Epsilon (Used in Adam, RMSProp to avoid division by zero). Defaults to 1e-9.
        """
        self.m1 = m1
        self.m2 = m2
        self.ep = ep

        self.vdW = cp.zeros(W_shape)
        self.vdb = cp.zeros(b_shape)

        self.SdW = cp.zeros(W_shape)
        self.Sdb = cp.zeros(b_shape)

    @abstractmethod
    def update(self, dW: npt.NDArray[Any], db: npt.NDArray[Any], k: int) -> tuple[npt.NDArray[Any]]:
        """Update parameters via optimisers

        Args:
            dW (npt.NDArray[Any]): Weight gradient
            db (npt.NDArray[Any]): Bias gradient
            k (int): Iteration

        Returns:
            tuple[npt.NDArray[Any]]: Changes to gradients
        """
        pass

class GD(Optimiser):
    def update(self, dW, db, k):
        return dW, db

class SGD(Optimiser):
    def update(self, dW, db, k):
        self.vdW = self.m1*self.vdW + (1-self.m1)*dW
        self.vdb = self.m1*self.vdb + (1-self.m1)*db

        return self.vdW, self.vdb

class RMSProp(Optimiser):
    def update(self, dW, db, k):
        self.SdW = self.m1 * self.SdW + (1 - self.m1) * (dW ** 2)
        self.Sdb = self.m1 * self.Sdb + (1 - self.m1) * (db ** 2)

        den_W = cp.sqrt(self.SdW) + self.ep
        den_b = cp.sqrt(self.Sdb) + self.ep

        return dW/den_W, db/den_b
    

class Adam(Optimiser): 
    def update(self, dW, db, k):
        self.vdW = self.m1 * self.vdW + (1 - self.m1) * dW
        self.vdb = self.m1 * self.vdb + (1 - self.m1) * db

        self.SdW = self.m2 * self.SdW + (1 - self.m2) * (dW ** 2)
        self.Sdb = self.m2 * self.Sdb + (1 - self.m2) * (db ** 2)

        if k > 1:
            den_m1 = (1 - (self.m1 ** k))
            den_m2 = (1 - (self.m2 ** k))
            vdW_h, vdb_h, SdW_h, Sdb_h = self.vdW / den_m1, self.vdb / den_m1, self.SdW / den_m2, self.Sdb / den_m2
        else:
            vdW_h, vdb_h, SdW_h, Sdb_h = self.vdW, self.vdb, self.SdW, self.Sdb

        den_W = cp.sqrt(SdW_h) + self.ep
        den_b = cp.sqrt(Sdb_h) + self.ep

        dW = vdW_h/den_W
        db = vdb_h/den_b

        return dW, db
