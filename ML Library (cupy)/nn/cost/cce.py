import cupy as cp
from nn.cost import Cost

class CCE(Cost):
    Ep = 1e-6
    def cost(self, A, Y):
        A = cp.clip(A, CCE.Ep, 1. - CCE.Ep)
        return -cp.sum(Y * cp.log(A))
    
    def backward(self, A, Y, clip_norm=0.5):
        A = cp.clip(A, CCE.Ep, 1. - CCE.Ep)
        grad = -Y/A
        return grad / cp.linalg.norm(grad) * clip_norm
    
    def predictions(self, A, Y):
        return cp.argmax(A, axis=1).tolist(), cp.argmax(Y, axis=1).tolist()
    
    def accuracy(self, pred, ans):
        return cp.mean(pred == ans)