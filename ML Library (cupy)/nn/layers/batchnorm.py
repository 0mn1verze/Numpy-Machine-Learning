import cupy as cp
from nn.layers import Layer

class BatchNorm(Layer):
    def __init__(self, m: float=0.9, ep: float=1e-6):
        """Batch Normalisation Layer

        Args:
            m (float, optional): Momentum for the moving average. Defaults to 0.9.
            ep (float, optional): Avoids dividing by zero. Defaults to 1e-6.
        """

        self.ep = ep
        self.m1 = m
    
    def setup(self, in_shape, optimiser):
        """Set up layer parameters

        Args:
            d (int): Input shape
        """
        self.in_shape = in_shape
        self.out_shape = in_shape
        self.gamma = cp.ones(self.in_shape)
        self.beta = cp.zeros(self.in_shape)
        self.r_mean = cp.zeros(self.in_shape)
        self.r_var = cp.zeros(self.in_shape)

    def forward(self, X, mode="train"):
        match mode:
            case "train":
                self.m, self.d = X.shape
                self.mu = cp.mean(X, axis=0)
                self.var = cp.var(X, axis=0)
                self.X_bar = X - self.mu
                self.i_var = 1 / cp.sqrt(self.var + self.ep)
                self.X_hat = self.X_bar * self.i_var
                q = self.gamma * self.X_hat + self.beta
                self.r_mean = self.m1 * self.r_mean + (1 - self.m1) * self.mu
                self.r_var = self.m1 * self.r_var + (1 - self.m1) * self.var
            case 'test':
                q = (X - self.r_mean) / cp.sqrt(self.r_var + self.ep)
                q = self.gamma*q + self.beta
            case other:
                raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
        return q
    
    def backward(self, dq):
        self.dgamma = cp.sum(dq * self.X_hat, axis=0)
        self.dbeta = cp.sum(dq, axis=0)
        dX_hat = dq * self.gamma
        dvar = cp.sum(dX_hat * self.X_bar * (-.5) * (self.i_var**3), axis=0)
        dmu = cp.sum(dX_hat * (-self.i_var), axis=0)
        dX = dX_hat * self.i_var + dvar * (2/self.m) * self.X_bar + (1/self.m)*dmu
        return dX
    
    def update(self, lr: float, m: int, k: int):
        """Update Parameters

        Args:
            lr (float): Learning rate
            m (int): Batch size
            k (int): Iteration
        """

        self.gamma -= self.dgamma * lr / m
        self.beta -= self.dbeta * lr / m

    def params(self):
        return 2 * self.in_shape, 2 * self.in_shape