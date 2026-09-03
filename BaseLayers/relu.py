try:
    import cupy as cp
except ImportError:
    import numpy as cp

class ReLU():        
    def forward(self, x):
        self.x = x
        return cp.maximum(0., x)

    def backward(self, grad):
        dx = grad * (self.x > 0).astype(self.x.dtype)
        del self.x
        return dx

class GELU():
    """
    Gaussian Error Linear Unit (standard ViT activation):
        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def forward(self, x):
        self.x = x
        self.cdf = 0.5 * (1.0 + cp.tanh(cp.sqrt(2.0 / cp.pi) * (x + 0.044715 * cp.power(x, 3))))
        return x * self.cdf

    def backward(self, grad):
        x = self.x
        tanh_out = cp.tanh(cp.sqrt(2.0 / cp.pi) * (x + 0.044715 * cp.power(x, 3)))
        sech2 = 1.0 - tanh_out ** 2
        d_tanh = cp.sqrt(2.0 / cp.pi) * (1.0 + 3.0 * 0.044715 * (x ** 2))
        d_cdf = 0.5 * sech2 * d_tanh
        dx = grad * (self.cdf + x * d_cdf)
        del self.x, self.cdf
        return dx

