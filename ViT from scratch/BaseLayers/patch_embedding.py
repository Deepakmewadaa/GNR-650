try:
    import cupy as cp
except ImportError:
    import numpy as cp

class PatchEmbedding():
    """
    Standard Vision Transformer Patch Embedding (implemented from scratch):
    Divides an image (B, C, H, W) into N non-overlapping PxP patches
    and linearly projects each patch to d_model embedding dimension:
        N = (H // P) * (W // P)
        Patch vector dimension = C * P * P
    """
    def __init__(self, img_size, patch_size, in_channels, d_model, optimizer, data_type=cp.float32):
        self.layer_name = "patch_embed"
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.optimizer = optimizer
        self.data_type = data_type
        
        self.grid_h = self.img_size[0] // self.patch_size[0]
        self.grid_w = self.img_size[1] // self.patch_size[1]
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = self.in_channels * self.patch_size[0] * self.patch_size[1]
        
        self.weights = None
        self.bias = None
        self.grad_weights = None
        self.grad_bias = None
        
        self.x = None
        self.patches = None
        
        self.init_weights()
        self.zero_grad()
        self.register()

    def init_weights(self):
        # Truncated normal / Xavier-like initialization for linear patch projection
        scale = cp.sqrt(2.0 / (self.patch_dim + self.d_model))
        self.weights = cp.random.normal(0, scale, (self.patch_dim, self.d_model)).astype(self.data_type)
        self.bias = cp.zeros((self.d_model,), dtype=self.data_type)

    def zero_grad(self):
        self.grad_weights = cp.zeros_like(self.weights)
        self.grad_bias = cp.zeros_like(self.bias)

    def register(self):
        cnt = self.optimizer.count_layers(f"{self.layer_name}_w")
        self.w_reg_name = f"{self.layer_name}_w_{cnt}"
        self.optimizer.register_params(self.w_reg_name, self.weights)
        
        cnt_b = self.optimizer.count_layers(f"{self.layer_name}_b")
        self.b_reg_name = f"{self.layer_name}_b_{cnt_b}"
        self.optimizer.register_params(self.b_reg_name, self.bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, num_patches, d_model)
        """
        self.x = cp.asarray(x, dtype=self.data_type)
        B, C, H, W = self.x.shape
        pH, pW = self.patch_size
        gH, gW = self.grid_h, self.grid_w
        
        # Reshape to (B, C, gH, pH, gW, pW) -> transpose to (B, gH, gW, C, pH, pW) -> (B, num_patches, patch_dim)
        x_reshaped = self.x.reshape(B, C, gH, pH, gW, pW)
        x_transposed = x_reshaped.transpose(0, 2, 4, 1, 3, 5)
        self.patches = x_transposed.reshape(B, self.num_patches, self.patch_dim)
        
        # Project patches: (B, N, patch_dim) @ (patch_dim, d_model) -> (B, N, d_model)
        out = self.patches @ self.weights + self.bias
        return out

    def backward(self, grad):
        """
        grad: (B, num_patches, d_model)
        Returns: grad_x (B, C, H, W)
        """
        grad = cp.asarray(grad, dtype=self.data_type)
        B = grad.shape[0]
        pH, pW = self.patch_size
        gH, gW = self.grid_h, self.grid_w
        C = self.in_channels
        
        patches_flat = self.patches.reshape(-1, self.patch_dim)
        grad_flat = grad.reshape(-1, self.d_model)
        
        self.grad_weights += patches_flat.T @ grad_flat
        self.grad_bias += cp.sum(grad_flat, axis=0)
        
        # Gradient wrt patch vectors: (B, N, patch_dim)
        grad_patches = grad @ self.weights.T
        
        # Inverse permutation: (B, gH, gW, C, pH, pW) -> (B, C, gH, pH, gW, pW) -> (B, C, H, W)
        grad_reshaped = grad_patches.reshape(B, gH, gW, C, pH, pW)
        grad_x = grad_reshaped.transpose(0, 3, 1, 4, 2, 5).reshape(B, C, self.img_size[0], self.img_size[1])
        return grad_x

    def release_memory(self):
        del self.x, self.patches

    def update_weights(self):
        self.weights = self.optimizer.update(self.weights, self.grad_weights, self.w_reg_name)
        self.bias = self.optimizer.update(self.bias, self.grad_bias, self.b_reg_name)
        self.zero_grad()
