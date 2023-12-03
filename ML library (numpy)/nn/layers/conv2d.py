import numpy as np
from nn.layers import Layer, Pad2D
from nn.initialisers import He, Weight

class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: tuple[int] | int, p: tuple[int] | int | str="valid", s: tuple[int] | int=(1, 1), 
                 use_bias: bool = True, initialiser: Weight=He(), regulariser: tuple[int]=('L2', 0), seed: int=None):
        """Convolutional 2D Layer

        Args:
            filters (int): the number of output filters
            kernel_size (tuple[int] | int): Kernel size
            s (tuple[int] | int, optional): Stride pattern. Defaults to (1, 1).
            use_bias (bool, optional): Determines if the layer uses bias. Defaults to True.
            initialiser (Weight, optional): Weight initialiser. Defaults to He.
            seed (int, optional): PRNG seed. Defaults to None.
            regulariser (tuple[str, int], optional): weight regulariser. Defaults to ('L2', 0)
        """
        self.pad = Pad2D(p=p)
        self.F = filters

        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        elif type(kernel_size) == tuple and len(kernel_size) == 2:
            self.kernel_size = kernel_size

        self.Kh, self.Kw = self.kernel_size

        if type(s) == int:
            self.s = (s, s)
        elif type(s) == tuple and len(s) == 2:
            self.s = s
        
        self.sh, self.sw = self.s

        self.use_bias = use_bias
        self.initialiser = initialiser

        self.regulariser = regulariser

        self.seed = seed
    
    def setup(self, in_shape, optimiser):
        self.in_shape_x = in_shape

        self.in_shape, _ = self.pad.get_dimensions(self.in_shape_x, self.kernel_size, self.s)

        if len(in_shape) == 3:  
            self.Nc, self.Nh, self.Nw = self.in_shape
        elif len(in_shape) == 4:
            self.m, self.Nc, self.Nh, self.Nw = self.in_shape
        
        self.Oh = (self.Nh - self.Kh) // self.sh + 1
        self.Ow = (self.Nw - self.Kw) // self.sw + 1

        if len(in_shape) == 3:
            self.out_shape = (self.F, self.Oh, self.Ow)
        elif len(in_shape) == 4:
            self.out_shape = (self.m, self.F, self.Oh, self.Ow)

        self.b_shape = (self.F, self.Oh, self.Ow)
        self.K_shape = (self.F, self.Nc, self.Kh, self.Kw)

        self.K = self.initialiser.initialise(self.K_shape, self.seed)
        self.b = np.zeros(self.b_shape)

        self.optimiser = optimiser(self.K_shape, self.b_shape)

    def params(self):
        return ((self.Nc * self.Kh * self.Kw) + 1 if self.use_bias else 0) * self.F, 0

    def __dilate2D(self, X, Dr=(1, 1)):
        dh, dw = Dr
        m, C, H, W = X.shape
        Xd = np.insert(arr=X, obj=np.repeat(np.arange(1, W), dw-1), values=0, axis=-1)
        Xd = np.insert(arr=Xd, obj=np.repeat(np.arange(1, H), dh-1), values=0, axis=-2)
        return Xd
    
    def __prepare_subM(self, X, Kh, Kw, s):
        m, Nc, Nh, Nw = X.shape
        sh, sw = s

        Oh = (Nh - Kh) // sh + 1
        Ow = (Nw - Kw) // sw + 1

        strides = (Nc*Nh*Nw, Nw*Nh, Nw*sh, sw, Nw, 1)
        strides = tuple(i * X.itemsize for i in strides)

        subM = np.lib.stride_tricks.as_strided(X, shape=(m, Nc, Oh, Ow, Kh, Kw), strides=strides)

        return subM
    
    def __convolve(self, X, K, s=(1,1), mode='forward'):
        F, Kc, Kh, Kw = K.shape
        subM = self.__prepare_subM(X, Kh, Kw, s)
        
        if mode=='forward':
            return np.einsum('fckl,mcijkl->mfij', K, subM)
        elif mode=='backward':
            return np.einsum('fdkl,mcijkl->mdij', K, subM)
        elif mode=='param':
            return np.einsum('mfkl,mcijkl->fcij', K, subM)
        
    def forward(self, X):
        self.X = X

        Xp = self.pad.forward(X, self.kernel_size, self.s)

        Z = self.__convolve(Xp, self.K, self.s) + self.b

        return Z
    
    def __dZ_D_dX(self, dZ_D, Nh, Nw):


        _, _, Hd, Wd = dZ_D.shape

        ph = Nh - Hd + self.Kh - 1
        pw = Nw - Wd + self.Kw - 1
        
        pad_back = Pad2D(p=(ph, pw))

        dZ_Dp = pad_back.forward(dZ_D, self.kernel_size, self.s)
       
        K_rotated = self.K[:, :, ::-1, ::-1]
                
        dXp = self.__convolve(dZ_Dp, K_rotated, mode='backward')
        
        dX = self.pad.backward(dXp)

        return dX
    
    def backward(self, dZ):

        Xp = self.pad.forward(self.X, self.kernel_size, self.s)

        m, Nc, Nh, Nw = Xp.shape
        
        dZ_D = self.__dilate2D(dZ, Dr=self.s)
        
        dX = self.__dZ_D_dX(dZ_D, Nh, Nw)
        
        _, _, Hd, Wd = dZ_D.shape
        
        ph = self.Nh - Hd - self.Kh + 1
        pw = self.Nw - Wd - self.Kw + 1

        pad_back = Pad2D(p=(ph, pw))

        dZ_Dp = pad_back.forward(dZ_D, self.kernel_size, self.s)
        
        self.dK = self.__convolve(Xp, dZ_Dp, mode='param')
        
        self.db = np.sum(dZ, axis=0)

        return dX
    
    def update(self, lr, m, k):
        dK, db = self.optimiser.update(self.dK, self.db, k)

        if self.regulariser[0].lower()=='l2':
            dK += self.regulariser[1] * self.K
        elif self.regulariser[0].lower()=='l1':
            dK += self.regulariser[1] * np.sign(self.K)
        
        self.K -= dK * lr / m

        if self.use_bias:
            self.b -= db * lr / m