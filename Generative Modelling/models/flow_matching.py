import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class FlowMatching(nn.Module):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) / Rectified Flow.
    Generates high-fidelity images via straight continuous-time probability paths:
      x_t = (1 - t) * x_0 + t * x_1
    Target velocity vector field:
      u_t = x_1 - x_0
    """
    def __init__(self, unet: nn.Module, text_encoder: nn.Module, sigma_min: float = 1e-4):
        super().__init__()
        self.unet = unet
        self.text_encoder = text_encoder
        self.sigma_min = sigma_min

    def compute_loss(
        self,
        x_1: torch.Tensor,
        token_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        uncond_prob: float = 0.1
    ) -> torch.Tensor:
        """
        Computes the OT-CFM regression loss:
          L = E || v_theta(x_t, t, c) - (x_1 - x_0) ||^2
        """
        B = x_1.shape[0]
        device = x_1.device
        
        # 1. Sample standard Gaussian noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # 2. Sample time uniformly t ~ U(0, 1)
        t = torch.rand(B, device=device)
        t_expanded = t[:, None, None, None]
        
        # 3. Straight linear interpolation trajectory
        x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
        
        # 4. Target constant velocity vector field
        target_v = x_1 - x_0
        
        # 5. Extract text embeddings with Classifier-Free Guidance (CFG) dropout
        hidden_text, pooled_text = self.text_encoder(token_ids, attn_mask)
        
        if uncond_prob > 0.0:
            drop_mask = (torch.rand(B, device=device) < uncond_prob)
            if drop_mask.any():
                hidden_text = hidden_text.clone()
                pooled_text = pooled_text.clone()
                hidden_text[drop_mask] = 0.0
                pooled_text[drop_mask] = 0.0
                
        # 6. Predict vector field v_theta
        pred_v = self.unet(x_t, t, hidden_text, pooled_text)
        
        # 7. MSE loss
        loss = F.mse_loss(pred_v, target_v)
        return loss

    @torch.inference_mode()
    def sample(
        self,
        token_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        num_steps: int = 14,
        cfg_scale: float = 4.5,
        image_size: int = 64
    ) -> torch.Tensor:
        """
        Integrates the learned velocity field from t=0 (Gaussian noise) to t=1 (generated image)
        using a batched, GPU-parallelized Euler ODE solver with Classifier-Free Guidance.
        """
        B = token_ids.shape[0]
        device = token_ids.device
        
        # 1. Compute conditional prompt embeddings
        c_hidden, c_pooled = self.text_encoder(token_ids, attn_mask)
        
        # 2. Unconditional null prompt representations
        u_hidden = torch.zeros_like(c_hidden)
        u_pooled = torch.zeros_like(c_pooled)
        
        # Concatenate for single-pass batched forward execution
        ctx_both = torch.cat([c_hidden, u_hidden], dim=0)
        p_ctx_both = torch.cat([c_pooled, u_pooled], dim=0)
        
        # 3. Initialize prior from standard normal distribution
        x = torch.randn(B, 3, image_size, image_size, device=device)
        
        dt = 1.0 / num_steps
        
        # 4. Fast Euler ODE integration
        for step in range(num_steps):
            t_val = step * dt
            t_both = torch.full((2 * B,), t_val, device=device)
            x_both = torch.cat([x, x], dim=0)
            
            # Batched forward pass through UNet
            v_both = self.unet(x_both, t_both, ctx_both, p_ctx_both)
            v_cond, v_uncond = v_both.chunk(2, dim=0)
            
            # Classifier-Free Guidance extrapolation
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            x = x + v * dt

        # Map from [-1, 1] to [0, 1]
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        return x
