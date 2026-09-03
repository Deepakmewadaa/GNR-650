try:
    import cupy as cp
except ImportError:
    import numpy as cp
from dropout import Dropout
from patch_embedding import PatchEmbedding

class ViTInputEmbedding():
    """
    Turns a batch of images into the token sequence fed to the encoder:
      1. Converts images into patch tokens using either:
         - 'fovi': FOVI's KNNPartitioningPatchEmbedding (using PyTorch bridge)
         - 'standard': Pure NumPy/CuPy PatchEmbedding
      2. Prepends learnable [CLS] token
      3. Adds learnable positional embedding
      4. Applies Dropout
    """

    def __init__(self, img_size, patch_size, in_channels, d_model, optimizer, 
                 tokenizer_type='fovi', dropout_rate=0.1, data_type=cp.float32):
        self.layer_name = "vit_input_embedding"
        self.optimizer = optimizer
        self.d_model = d_model
        self.data_type = data_type
        self.tokenizer_type = tokenizer_type
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        if self.tokenizer_type == 'fovi':
            import os
            import sys
            import torch
            _fovi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fovi"))
            if _fovi_path not in sys.path:
                sys.path.append(_fovi_path)
            os.environ['FOVI_SAVE_DIR'] = './'
            os.environ['FOVI_DATASETS_DIR'] = './'
            from fovi.arch.knnvit import KNNPartitioningPatchEmbedding
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.patch_embed = KNNPartitioningPatchEmbedding(
                in_channels=in_channels,
                embed_dim=d_model,
                in_res=img_size,
                fov=20,
                cmf_a=1.0,
                style='isotropic',
                in_cart_res=img_size,
                cart_patch_size=patch_size,
                device=self.device,
                sample_cortex='none',
                transposed=False
            )
            self.patch_embed.to(self.device)
            self.num_patches = self.patch_embed.out_coords.shape[0]
            self.torch_optimizer = torch.optim.Adam(self.patch_embed.parameters(), lr=1e-3)
            self.x_torch = None
            self.output_torch = None
        else:
            self.patch_embed = PatchEmbedding(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                d_model=d_model,
                optimizer=optimizer,
                data_type=data_type
            )
            self.num_patches = self.patch_embed.num_patches

        self.seq_len = self.num_patches + 1  # +1 for CLS token
        self.dropout = Dropout(dropout_rate, data_type)

        self.cls_token = None
        self.pos_embedding = None
        self.grad_cls_token = None
        self.grad_pos_embedding = None

        self.init_weights()
        self.zero_grad()
        self.register()

    def init_weights(self):
        self.cls_token = cp.random.normal(0, 0.02, (1, 1, self.d_model)).astype(self.data_type)
        self.pos_embedding = cp.random.normal(0, 0.02, (1, self.seq_len, self.d_model)).astype(self.data_type)

    def zero_grad(self):
        self.grad_cls_token = cp.zeros_like(self.cls_token)
        self.grad_pos_embedding = cp.zeros_like(self.pos_embedding)

    def register(self):
        cnt = self.optimizer.count_layers(f'{self.layer_name}_cls')
        self.cls_registered_name = f'{self.layer_name}_cls_{cnt}'
        self.optimizer.register_params(self.cls_registered_name, self.cls_token)

        cnt_pos = self.optimizer.count_layers(f'{self.layer_name}_pos')
        self.pos_registered_name = f'{self.layer_name}_pos_{cnt_pos}'
        self.optimizer.register_params(self.pos_registered_name, self.pos_embedding)

    def forward(self, x, training=True):
        self.batch_size = x.shape[0]

        if self.tokenizer_type == 'fovi':
            import torch
            if hasattr(x, '__dlpack__') and str(cp.__name__) == 'cupy':
                try:
                    self.x_torch = torch.from_dlpack(x).to(self.device)
                except Exception:
                    self.x_torch = torch.tensor(cp.asnumpy(x), device=self.device, dtype=torch.float32)
            else:
                x_np = cp.asnumpy(x) if hasattr(cp, 'asnumpy') else x
                self.x_torch = torch.tensor(x_np, device=self.device, dtype=torch.float32)

            if training:
                self.x_torch.requires_grad = True
                self.output_torch = self.patch_embed(self.x_torch)
                if self.device == 'cuda' and str(cp.__name__) == 'cupy':
                    patches = cp.from_dlpack(self.output_torch.detach())
                else:
                    patches = cp.array(self.output_torch.detach().cpu().numpy())
            else:
                with torch.no_grad():
                    out = self.patch_embed(self.x_torch)
                    if self.device == 'cuda' and str(cp.__name__) == 'cupy':
                        patches = cp.from_dlpack(out)
                    else:
                        patches = cp.array(out.cpu().numpy())
        else:
            patches = self.patch_embed.forward(x)

        cls_tokens = cp.tile(self.cls_token, (self.batch_size, 1, 1))
        tokens = cp.concatenate([cls_tokens, patches], axis=1)
        tokens = tokens + self.pos_embedding
        output = self.dropout.forward(tokens, training)
        return output

    def backward(self, grad):
        grad = self.dropout.backward(grad)

        self.grad_pos_embedding += cp.sum(grad, axis=0, keepdims=True)

        grad_cls = grad[:, 0:1, :]
        grad_patches = grad[:, 1:, :]

        self.grad_cls_token += cp.sum(grad_cls, axis=0, keepdims=True)

        if self.tokenizer_type == 'fovi':
            import torch
            if hasattr(grad_patches, '__dlpack__') and str(cp.__name__) == 'cupy':
                try:
                    grad_patches_torch = torch.from_dlpack(grad_patches).to(self.device)
                except Exception:
                    grad_patches_torch = torch.tensor(cp.asnumpy(grad_patches), device=self.device, dtype=torch.float32)
            else:
                gp_np = cp.asnumpy(grad_patches) if hasattr(cp, 'asnumpy') else grad_patches
                grad_patches_torch = torch.tensor(gp_np, device=self.device, dtype=torch.float32)

            self.output_torch.backward(grad_patches_torch)

            if self.x_torch.grad is not None:
                if self.device == 'cuda' and str(cp.__name__) == 'cupy':
                    grad_images = cp.from_dlpack(self.x_torch.grad)
                else:
                    grad_images = cp.array(self.x_torch.grad.cpu().numpy())
            else:
                grad_images = None
        else:
            grad_images = self.patch_embed.backward(grad_patches)

        return grad_images

    def update_weights(self):
        self.cls_token = self.optimizer.update(self.cls_token, self.grad_cls_token, self.cls_registered_name)
        self.pos_embedding = self.optimizer.update(self.pos_embedding, self.grad_pos_embedding, self.pos_registered_name)
        
        if self.tokenizer_type == 'fovi':
            self.torch_optimizer.step()
            self.torch_optimizer.zero_grad()
        else:
            self.patch_embed.update_weights()
            
        self.zero_grad()

