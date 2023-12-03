import cupy as cp
from nn.layers import Layer, Pad2D

class Pool2D(Layer):
    def __init__(self, pool_size: tuple[int] | int=(2,2), p: tuple[int] | int | str ="valid", s: tuple[int]=(2,2), pool_type: str="max"):
        """Max Pooling 2D Layer

        Args:
            pool_size (tuple[int], optional): Pool size. Defaults to (2, 2).
            p (tuple[int] | int | str, optional): Padding amount/mode. Defaults to "valid".
            s (tuple[int], optional): Stride pattern. Defaults to (2, 2).
            pool_type (str, optional): Pooling type. Defaults to max.
        """

        self.pad = Pad2D(p=p)

        if type(pool_size) == int:
            self.pool_size = (pool_size, pool_size)
        elif type(pool_size) == tuple and len(pool_size) == 2:
            self.pool_size = pool_size

        self.Kh, self.Kw = self.pool_size

        if type(s) == int:
            self.s = (s, s)
        elif type(s) == tuple and len(s) == 2:
            self.s = s
        
        self.sh, self.sw = self.s

        self.pool_type = pool_type

    def setup(self, in_shape: tuple[int]):

        self.in_shape = in_shape

        in_shape, _ = self.pad.get_dimensions(in_shape, self.pool_size, self.s)

        if len(in_shape) == 4:
            m, Nc, Nh, Nw = in_shape
        elif len(in_shape) == 3:
            Nc, Nh, Nw = in_shape
        
        Oh = (Nh - self.Kh) // self.sh + 1
        Ow = (Nw - self.Kw) // self.sw + 1

        if len(in_shape) == 4:
            self.out_shape = (m, Nc, Oh, Ow)
        elif len(in_shape) == 3:
            self.out_shape = (Nc, Oh, Ow)

    def params(self):
        return 0, 0

    def __to_view(self, X, pool_size, s):
        m, Nc, Nh, Nw = X.shape
        Kh, Kw = pool_size
        sh, sw = s

        Oh = (Nh - Kh) // sh + 1
        Ow = (Nw - Kw) // sw + 1

        out_shape = (m, Nc, Oh, Ow, Kh, Kw)

        strides = (Nc*Nh*Nw, Nh*Nw, sh*Nw, sw, Nw, 1)
        strides = tuple(i * X.itemsize for i in strides)

        out = cp.lib.stride_tricks.as_strided(X, out_shape, strides)
        return out
    
    def __pooling(self, X, pool_size=(2,2), s=(2,2)):

        view = self.__to_view(X, pool_size, s)

        if self.pool_type == "max":
            return cp.nanmax(view, axis=(-2, -1))
        elif self.pool_type == "mean":
            return cp.nanmean(view, axis=(-2, -1))
        else:
            raise ValueError("Pool types allowed are: 'max' and 'mean'.")
        
    def __expand(self, view, dZ, Xp):

        m, Nc, Oh, Ow, Kh, Kw = view.shape

        view_size = m * Nc * Oh * Ow * Kh * Kw

        flatten = view.reshape(-1, Kh * Kw)

        max_idx = cp.argmax(flatten, axis=1) + cp.arange(0, view_size , Kh * Kw)

        unravelled = cp.unravel_index(max_idx, view.shape)


        dXp_view = self.__to_view(cp.zeros_like(Xp), self.pool_size, self.s)

        cp.add.at(dXp_view, unravelled, dZ.flatten())

        return cp.lib.stride_tricks.as_strided(dXp_view, Xp.shape, Xp.strides)

    def __maxpool_backward(self, dZ, X):
        Xp = self.pad.forward(X, self.pool_size, self.s)

        view = self.__to_view(Xp, self.pool_size, self.s)
        
        dXp = self.__expand(view, dZ, Xp)

        return dXp
    
    def __dZ_dZp(self, dZ):
        sh, sw = self.s
        Kh, Kw = self.pool_size

        dZp = cp.kron(dZ, cp.ones((Kh, Kw), dtype=dZ.dtype))

        jh, jw = Kh - sh, Kw - sw

        if jw!=0:
            L = dZp.shape[-1]-1

            l1 = cp.arange(sw, L)
            l2 = cp.arange(sw + jw, L + jw)

            mask = cp.tile([True]*jw + [False]*jw, len(l1)//jw).astype(bool)

            r1 = l1[mask[:len(l1)]]
            r2 = l2[mask[:len(l2)]]

            dZp[:, :, :, r1] += dZp[:, :, :, r2]
            dZp = cp.delete(dZp, r2, axis=-1)

        if jh!=0:
            L = dZp.shape[-2]-1

            l1 = cp.arange(sh, L)
            l2 = cp.arange(sh + jh, L + jh)

            mask = cp.tile([True]*jh + [False]*jh, len(l1)//jh).astype(bool)

            r1 = l1[mask[:len(l1)]]
            r2 = l2[mask[:len(l2)]]

            dZp[:, :, r1, :] += dZp[:, :, r2, :]
            dZp = cp.delete(dZp, r2, axis=-2)

        return dZp
    
    def __meanpool_backward(self, dZ, X):

        Xp = self.pad.forward(X, self.pool_size, self.s)

        m, Nc, Nh, Nw = Xp.shape

        dZp = self.__dZ_dZp(dZ)

        ph = Nh - dZp.shape[-2]
        pw = Nw - dZp.shape[-1]

        pad_back = Pad2D(p=(ph,pw))

        dXp = pad_back.forward(dZp, self.pool_size, self.s)

        return dXp / (Nh * Nw)
    
    def forward(self, X):
        self.X = X

        Xp = self.pad.forward(X, self.pool_size, self.s)

        Z = self.__pooling(Xp, self.pool_size, self.s)

        return Z
    
    def backward(self, dZ):

        if self.pool_type == "max":
            dXp = self.__maxpool_backward(dZ, self.X)
        elif self.pool_type == "mean":
            dXp = self.__meanpool_backward(dZ, self.X)
        dX = self.pad.backward(dXp)

        return dX


