import cupy as cp
from nn.cost import Cost

class MSE(Cost):
    def cost(self, A, Y):
        return (1/2)*cp.sum(cp.linalg.norm(A-Y, axis=1)**2)
    
    def backward(self, A, Y):
        return A - Y
    
    def predictions(self, A, Y):
        return cp.argmax(A, axis=1).tolist(), cp.argmax(Y, axis=1).tolist()
    
    def accuracy(self, pred, ans):
        return cp.mean(pred == ans)