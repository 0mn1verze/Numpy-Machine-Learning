import numpy as np
from nn.cost import Cost

class CCE(Cost):
    Ep = 1e-6
    def cost(self, A, Y):
        A = np.clip(A, CCE.Ep, 1. - CCE.Ep)
        return -np.sum(Y * np.log(A))
    
    def backward(self, A, Y, clip_norm=0.5):
        A = np.clip(A, CCE.Ep, 1. - CCE.Ep)
        grad = -Y/A
        return grad / np.linalg.norm(grad) * clip_norm
    
    def predictions(self, A, Y):
        return np.argmax(A, axis=1).tolist(), np.argmax(Y, axis=1).tolist()
    
    def accuracy(self, pred, ans):
        return np.mean(pred == ans)