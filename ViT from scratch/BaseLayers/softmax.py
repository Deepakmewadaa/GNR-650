try:
    import cupy as cp
except ImportError:
    import numpy as cp

class Softmax():
    def __init__(self, axis=-1):
        self.axis = axis
        self.y = None

    def forward(self, x):
        shifted_x = x - cp.max(x, axis=self.axis, keepdims=True)
        e_x = cp.exp(shifted_x)
        self.y = e_x / (cp.sum(e_x, axis=self.axis, keepdims=True) + 1e-12)
        return self.y

    def backward(self, grad_y):
        grad_x = self.y * (grad_y - cp.sum(grad_y * self.y, axis=self.axis, keepdims=True))
        return grad_x


