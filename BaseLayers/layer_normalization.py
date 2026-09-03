try:
    import cupy as cp
except ImportError:
    import numpy as cp

class LayerNormalization():
    """
    Standard Layer Normalization layer:
        y = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    def __init__(self, optimizer, normalized_shape, eps=1e-05, data_type=cp.float32):
        self.layer_name = "layernorm"
        self.optimizer = optimizer
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.data_type = data_type
        
        self.gamma = None
        self.beta = None
        self.grad_gamma = None
        self.grad_beta = None
        
        self.x = None
        self.mean = None
        self.var = None
        self.std = None
        self.x_hat = None
        
        self.init_weights()
        self.zero_grad()
        self.register()

    def init_weights(self):
        self.gamma = cp.ones((self.normalized_shape,), dtype=self.data_type)
        self.beta = cp.zeros((self.normalized_shape,), dtype=self.data_type)

    def zero_grad(self):
        self.grad_gamma = cp.zeros_like(self.gamma)
        self.grad_beta = cp.zeros_like(self.beta)

    def register(self):
        self.layer_id = self.optimizer.count_layers(self.layer_name)
        self.register_name = "{}_{}".format(self.layer_name, self.layer_id)
        self.optimizer.register_params("{}.gamma".format(self.register_name), self.gamma)
        self.optimizer.register_params("{}.beta".format(self.register_name), self.beta)

    def forward(self, x):
        self.x = cp.asarray(x, dtype=self.data_type)
        self.mean = cp.mean(self.x, axis=-1, keepdims=True)
        self.var = cp.var(self.x, axis=-1, keepdims=True)
        self.std = cp.sqrt(self.var + self.eps)
        self.x_hat = (self.x - self.mean) / self.std
        y = self.gamma * self.x_hat + self.beta
        return y

    def backward(self, grad):
        grad = cp.asarray(grad, dtype=self.data_type)
        reduce_axes = tuple(range(grad.ndim - 1))
        self.grad_gamma += cp.sum(grad * self.x_hat, axis=reduce_axes)
        self.grad_beta += cp.sum(grad, axis=reduce_axes)

        dx_hat = grad * self.gamma
        # Efficient LayerNorm backward formula
        mean_dx_hat = cp.mean(dx_hat, axis=-1, keepdims=True)
        mean_dx_hat_x_hat = cp.mean(dx_hat * self.x_hat, axis=-1, keepdims=True)
        dx = (dx_hat - mean_dx_hat - self.x_hat * mean_dx_hat_x_hat) / self.std
        return dx

    def release_memory(self):
        del self.x, self.mean, self.var, self.std, self.x_hat

    def update_weights(self):
        self.gamma = self.optimizer.update(self.gamma, self.grad_gamma, "{}.gamma".format(self.register_name))
        self.beta = self.optimizer.update(self.beta, self.grad_beta, "{}.beta".format(self.register_name))
        self.zero_grad()
