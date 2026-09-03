
try:
    import cupy as cp
except ImportError:
    import numpy as cp
from utils import _release_memory
class Linear():
    def __init__(self, in_features, out_features, optimizer, use_bias=True, data_type=cp.float32):
        self.layer_name = "linear"
        self.in_features = in_features
        self.out_features = out_features
        self.optimizer = optimizer
        self.use_bias = use_bias
        self.weights = None
        self.bias= None
        self.grad_weights = None
        self.grad_bias = None
        self.data_type = data_type
        self.init_weights()
        self.zero_grad()
        self.register()


    def init_weights(self):
        sqrt_k = 1. / cp.sqrt(self.in_features)
        self.weights = cp.random.uniform(-sqrt_k, sqrt_k, (self.in_features, self.out_features)).astype(self.data_type)
        if self.use_bias:
            self.bias = cp.random.uniform(-sqrt_k, sqrt_k, self.out_features).astype(self.data_type)

    def zero_grad(self):
        self.grad_weights = cp.zeros_like(self.weights)
        if self.use_bias:
            self.grad_bias = cp.zeros_like(self.bias)
    
    def register(self):
        weights_registered_name = '{}_{}'.format(self.layer_name, 'weights')
        cnt= self.optimizer.count_layers(weights_registered_name)
        self.weights_registered_name = "{}_{}".format(weights_registered_name, cnt)
        self.optimizer.register_params(self.weights_registered_name, self.weights)
        if self.use_bias:
            bias_registered_name = '{}_{}'.format(self.layer_name, 'bias')
            cnt= self.optimizer.count_layers(bias_registered_name)
            self.bias_registered_name = "{}_{}".format(bias_registered_name, cnt)
            self.optimizer.register_params(self.bias_registered_name, self.bias)

    def forward(self, x):
        self.x = cp.asarray(x, dtype=self.data_type)
        self.output = self.x @ self.weights
        if self.use_bias:
            self.output = self.output + self.bias
        return self.output
    
    def backward(self, grad):
        grad = cp.asarray(grad, dtype=self.data_type)
        if self.x.ndim == 2:
            self.grad_weights += self.x.T @ grad
            if self.use_bias:
                self.grad_bias += cp.sum(grad, axis=0)
        elif self.x.ndim == 3:
            x_flat = self.x.reshape(-1, self.in_features)
            grad_flat = grad.reshape(-1, self.out_features)
            self.grad_weights += x_flat.T @ grad_flat
            if self.use_bias:
                self.grad_bias += cp.sum(grad, axis=(0, 1))
        else:
            x_flat = self.x.reshape(-1, self.in_features)
            grad_flat = grad.reshape(-1, self.out_features)
            self.grad_weights += x_flat.T @ grad_flat
            if self.use_bias:
                self.grad_bias += cp.sum(grad, axis=tuple(range(grad.ndim - 1)))
        self.grad = grad @ self.weights.T
        return self.grad
    
    def release_memory(self):
        del self.x, self.output
       

    def update_weights(self):
        self.weights = self.optimizer.update(self.weights, self.grad_weights, self.weights_registered_name)
        if self.use_bias:
            self.bias = self.optimizer.update(self.bias, self.grad_bias, self.bias_registered_name)
        self.release_memory()
        self.zero_grad()
