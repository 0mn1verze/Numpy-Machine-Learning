import numpy as np
from nn.cost import Cost

class MSE(Cost):
    def cost(self, A, Y):
        return (1/2)*np.sum(np.linalg.norm(A-Y, axis=1)**2)
    
    def backward(self, A, Y):
        return A - Y
    
    def predictions(self, A, Y):
        return np.argmax(A, axis=1).tolist(), np.argmax(Y, axis=1).tolist()
    
    def accuracy(self, pred, ans):
        return np.mean(pred == ans)