from nn.layers import Layer

class Flatten(Layer):
    def forward(self, X):
        self.m, self.Nc, self.Nh, self.Nw = X.shape
        X_flat = X.reshape((self.m, self.Nc * self.Nh * self.Nw))
        return X_flat
    
    def backward(self, dZ):
        dX = dZ.reshape((self.m, self.Nc, self.Nh, self.Nw))
        return dX
    
    def setup(self, in_shape):
        self.in_shape = in_shape
        if len(self.in_shape) == 4:
            self.m, self.Nc, self.Nh, self.Nw = self.in_shape
        elif len(self.in_shape) == 3:
            self.Nc, self.Nh, self.Nw = self.in_shape
        
        self.out_shape = self.Nc * self.Nh * self.Nw
    
    def params(self):
        return 0, 0

