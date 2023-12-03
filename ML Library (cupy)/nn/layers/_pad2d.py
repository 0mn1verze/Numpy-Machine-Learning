import cupy as cp
import numpy.typing as npt
from typing import Any

class Pad2D():
    def __init__(self, p: str | int | tuple ="valid"):
        """Pad2D

        Args:
            p (str | int | tuple, optional): Padding amount/mode. Defaults to "valid".
        """
        self.p = p
        
    def params(self):
        return 0, 0

    def get_dimensions(self, in_shape: tuple[int], kernel_size: tuple[int], s: tuple[int]=(1, 1)) -> tuple[int]:
        """Get padding dimensions

        Args:
            in_shape (tuple[int]): Input shape
            kernel_size (tuple[int]): Kernel size
            s (tuple[int], optional): Stride pattern. defaults to (1, 1)

        Raises:
            ValueError: If p is not an int or "valid" or "same", then the error is raised

        Returns:
            tuple[int]: Output shape
        """

        if len(in_shape) == 4:
            m, Nc, Nh, Nw = in_shape
        elif len(in_shape) == 3:
            Nc, Nh, Nw = in_shape
        
        Kh, Kw = kernel_size
        sh, sw = s
        p = self.p

        if type(p) == int:
            pt, pb = p, p
            pl, pr = p, p
        elif type(p) == tuple and len(p) == 2:
            ph, pw = p
            pt, pb = ph // 2, (ph + 1) // 2
            pl, pr = pw // 2, (pw + 1) // 2
        elif p == "valid":
            pt, pb = 0, 0
            pl, pr = 0, 0
        elif p == "same":
            ph = (sh - 1)*Nh + Kh - sh
            pw = (sw - 1)*Nw + Kw - sw

            pt, pb = ph // 2, (ph + 1) // 2
            pl, pr = pw // 2, (pw + 1) // 2

        else:
            raise ValueError("Padding type must be an int or 'same' or 'valid'.")

        if len(in_shape) == 4:
            output_shape = (m, Nc, Nh+pt+pb, Nw+pl+pr)
        elif len(in_shape) == 3:
            output_shape = (Nc, Nh+pt+pb, Nw+pl+pr)

        return output_shape, (pt, pb, pl, pr)
    
    def forward(self, X: npt.NDArray[Any], kernel_size: tuple[int], s: tuple[int]=(1, 1)) -> npt.NDArray[Any]:
        """Padding Forward Propagation

        Args:
            X (npt.NDArray[Any]): Input
            kernel_size (tuple[int]): Kernel size
            s (tuple[int], optional): Stride pattern. Defaults to (1, 1).

        Returns:
            npt.NDArray[Any]: Padded input
        """
        self.in_shape = X.shape
        m, Nc, Nh, Nw = self.in_shape

        self.out_shape, (self.pt, self.pb, self.pl, self.pr) = self.get_dimensions(self.in_shape, kernel_size, s)

        zeros_t = cp.zeros((m, Nc, self.pt, Nw + self.pl + self.pr))
        zeros_b = cp.zeros((m, Nc, self.pb, Nw + self.pl + self.pr))
        zeros_l = cp.zeros((m, Nc, Nh, self.pl))
        zeros_r = cp.zeros((m, Nc, Nh, self.pr))

        Xp = cp.concatenate((X, zeros_r), axis=3)
        Xp = cp.concatenate((zeros_l, Xp), axis=3)
        Xp = cp.concatenate((zeros_t, Xp), axis=2)
        Xp = cp.concatenate((Xp, zeros_b), axis=2)

        return Xp
    
    def backward(self, dXp: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Padding Backward Propagation

        Args:
            dXp (npt.NDArray[Any]): Gradient from next layer

        Returns:
            npt.NDArray[Any]: Gradient for previous layer
        """

        _, _, Nh, Nw = self.in_shape
        dX = dXp[:, :, self.pt:self.pt+Nh, self.pl:self.pl+Nw]

        return dX
    
    def params(self):
        return 0, 0
