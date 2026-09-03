try:
    import cupy as cp
except ImportError:
    import numpy as cp

import math

class Adam():
    """
    Adam Optimizer with Cosine Annealing, linear warmup, and state checkpointing.
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4, 
                 warmup_steps=0, total_steps=10000, min_lr=1e-5, scheduler='cosine', d_model=512):
        self.initial_lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step = 0
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler = scheduler
        self.d_model = d_model
        self.registered_layer_params = {}

    def _step(self):
        self.step += 1
        self.set_lr()

    def set_lr(self):
        if self.warmup_steps and self.step < self.warmup_steps:
            self.lr = self.initial_lr * float(self.step + 1) / float(self.warmup_steps)
            return

        if self.scheduler == 'cosine':
            effective_total = max(1, self.total_steps - self.warmup_steps)
            effective_step = max(0, self.step - self.warmup_steps)
            progress = min(1.0, float(effective_step) / float(effective_total))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            self.lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        elif self.scheduler == 'inv_sqrt':
            if self.warmup_steps > 0:
                self.lr = self.initial_lr * ((float(self.warmup_steps) / float(self.step + 1)) ** 0.5)
            else:
                self.lr = self.initial_lr / ((float(self.step + 1)) ** 0.5)
        else:  # 'constant'
            self.lr = self.initial_lr

    def save_params(self, params_name, t, mean, var):
        self.registered_layer_params[params_name] = {
            "t": t,
            "mean": mean,
            "var": var
        }

    def register_params(self, params_name, params):
        if params_name not in self.registered_layer_params:
            self.save_params(params_name=params_name, t=0, mean=cp.zeros_like(params), var=cp.zeros_like(params))
        else:
            # Re-register if shape changed
            if self.registered_layer_params[params_name]["mean"].shape != params.shape:
                self.save_params(params_name=params_name, t=0, mean=cp.zeros_like(params), var=cp.zeros_like(params))

    def count_layers(self, layer_name):
        cnt = 0
        for key in self.registered_layer_params.keys():
            if key.startswith(layer_name):
                cnt += 1
        return cnt

    def reset(self):
        self.step = 0
        self.registered_layer_params = {}

    def update(self, param, param_grad, params_name):
        assert param.shape == param_grad.shape, f'param shape: {param.shape}, param_grad shape: {param_grad.shape}'
        
        if params_name not in self.registered_layer_params:
            self.register_params(params_name, param)
            
        t = self.registered_layer_params[params_name]["t"]
        mean = self.registered_layer_params[params_name]["mean"]
        var = self.registered_layer_params[params_name]["var"]
        
        t += 1
        
        # Apply weight decay if configured (AdamW style)
        if self.weight_decay > 0.0:
            param = param - self.lr * self.weight_decay * param
            
        mean = self.beta1 * mean + (1.0 - self.beta1) * param_grad
        var = self.beta2 * var + (1.0 - self.beta2) * (param_grad ** 2)
        
        self.save_params(params_name, t, mean, var)
        
        mean_hat = mean / (1.0 - (self.beta1 ** t))
        var_hat = var / (1.0 - (self.beta2 ** t))
        
        delta = self.lr * mean_hat / (cp.sqrt(var_hat) + self.eps)
        param = param - delta
        return param

    def get_state(self):
        state = {
            'step': self.step,
            'initial_lr': self.initial_lr,
            'lr': self.lr,
            'params': {}
        }
        for k, v in self.registered_layer_params.items():
            state['params'][k] = {
                't': v['t'],
                'mean': cp.asnumpy(v['mean']) if hasattr(cp, 'asnumpy') else v['mean'],
                'var': cp.asnumpy(v['var']) if hasattr(cp, 'asnumpy') else v['var']
            }
        return state

    def set_state(self, state):
        self.step = state.get('step', 0)
        self.initial_lr = state.get('initial_lr', self.initial_lr)
        self.lr = state.get('lr', self.lr)
        if 'params' in state:
            for k, v in state['params'].items():
                self.registered_layer_params[k] = {
                    't': v['t'],
                    'mean': cp.array(v['mean']),
                    'var': cp.array(v['var'])
                }

class SGD():
    def __init__(self, lr=1e-3, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def update(self, param, param_grad, params_name="param"):
        if self.momentum > 0:
            if params_name not in self.velocities:
                self.velocities[params_name] = cp.zeros_like(param)
            v = self.momentum * self.velocities[params_name] - self.lr * param_grad
            self.velocities[params_name] = v
            return param + v
        return param - self.lr * param_grad

