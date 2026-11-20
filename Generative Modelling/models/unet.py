import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion/flow matching continuous time t in [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ResnetBlock(nn.Module):
    """
    Residual block with AdaLN / Scale-Shift modulation from combined
    time and pooled text embeddings.
    """
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        num_groups_in = min(32, in_channels)
        while in_channels % num_groups_in != 0:
            num_groups_in //= 2
            
        num_groups_out = min(32, out_channels)
        while out_channels % num_groups_out != 0:
            num_groups_out //= 2
            
        self.norm1 = nn.GroupNorm(num_groups=num_groups_in, num_channels=in_channels, eps=1e-6)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2) # Scale and shift
        )
        
        self.norm2 = nn.GroupNorm(num_groups=num_groups_out, num_channels=out_channels, eps=1e-6)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        
        scale_shift = self.cond_proj(cond_emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(self.act2(h)))
        return h + self.shortcut(x)

class MultiHeadCrossAttention(nn.Module):
    """
    Spatial Multi-Head Cross-Attention layer.
    Attends spatial image features (queries) to textual token embeddings (keys & values).
    """
    def __init__(self, query_dim: int, context_dim: int = 256, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        if context is None:
            context = x_flat # Self-attention fallback
            
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.view(B, -1, self.num_heads, q.shape[-1] // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, k.shape[-1] // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, v.shape[-1] // self.num_heads).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = scores.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, -1)
        out = self.to_out(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out

class SpatialTransformer(nn.Module):
    """
    Transformer block containing Self-Attention, Cross-Attention (with text), and FFN.
    """
    def __init__(self, in_channels: int, context_dim: int = 256, num_heads: int = 4, head_dim: int = 48):
        super().__init__()
        num_groups = min(32, in_channels)
        while in_channels % num_groups != 0:
            num_groups //= 2
            
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6)
        self.self_attn = MultiHeadCrossAttention(in_channels, in_channels, num_heads, head_dim)
        self.cross_attn = MultiHeadCrossAttention(in_channels, context_dim, num_heads, head_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )
        self.norm_ffn = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm(x)
        x = x + self.self_attn(norm_x)
        x = x + self.cross_attn(self.norm(x), context)
        x = x + self.ffn(self.norm_ffn(x))
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, has_attn: bool = False, context_dim: int = 256):
        super().__init__()
        self.res1 = ResnetBlock(in_channels, out_channels, cond_dim)
        self.res2 = ResnetBlock(out_channels, out_channels, cond_dim)
        self.has_attn = has_attn
        if has_attn:
            self.attn = SpatialTransformer(out_channels, context_dim=context_dim, num_heads=4, head_dim=out_channels // 4)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor, context: torch.Tensor):
        x = self.res1(x, cond_emb)
        x = self.res2(x, cond_emb)
        if self.has_attn:
            x = self.attn(x, context)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, cond_dim: int, has_attn: bool = False, context_dim: int = 256):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res1 = ResnetBlock(in_channels + skip_channels, out_channels, cond_dim)
        self.res2 = ResnetBlock(out_channels, out_channels, cond_dim)
        self.has_attn = has_attn
        if has_attn:
            self.attn = SpatialTransformer(out_channels, context_dim=context_dim, num_heads=4, head_dim=out_channels // 4)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond_emb: torch.Tensor, context: torch.Tensor):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, cond_emb)
        x = self.res2(x, cond_emb)
        if self.has_attn:
            x = self.attn(x, context)
        return x

class TextConditionedUNet(nn.Module):
    """
    High-capacity Text-Conditioned U-Net engineered to strictly satisfy:
      1. Trainable parameter count < 50M (~33M UNet + ~4.2M TextEncoder = ~37.2M total).
      2. Sub-second generation latency via efficient channel scaling and multi-head cross-attention.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ch_mult: list = [64, 128, 256, 384],
        context_dim: int = 256
    ):
        super().__init__()
        time_dim = 256
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim * 2)
        )
        self.text_pooled_proj = nn.Sequential(
            nn.Linear(context_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim * 2)
        )
        cond_dim = time_dim * 2
        
        self.conv_in = nn.Conv2d(in_channels, ch_mult[0], kernel_size=3, padding=1)
        
        # Encoder downsampling stages: 64 -> 32 -> 16 -> 8
        self.down1 = DownBlock(ch_mult[0], ch_mult[1], cond_dim, has_attn=False, context_dim=context_dim)
        self.down2 = DownBlock(ch_mult[1], ch_mult[2], cond_dim, has_attn=True, context_dim=context_dim)
        self.down3 = DownBlock(ch_mult[2], ch_mult[3], cond_dim, has_attn=True, context_dim=context_dim)
        
        # Bottleneck stage at 8x8 resolution
        self.mid_res1 = ResnetBlock(ch_mult[3], ch_mult[3], cond_dim)
        self.mid_attn = SpatialTransformer(ch_mult[3], context_dim=context_dim, num_heads=6, head_dim=ch_mult[3] // 6)
        self.mid_res2 = ResnetBlock(ch_mult[3], ch_mult[3], cond_dim)
        
        # Decoder upsampling stages: 8 -> 16 -> 32 -> 64
        self.up3 = UpBlock(ch_mult[3], ch_mult[3], ch_mult[2], cond_dim, has_attn=True, context_dim=context_dim)
        self.up2 = UpBlock(ch_mult[2], ch_mult[2], ch_mult[1], cond_dim, has_attn=True, context_dim=context_dim)
        self.up1 = UpBlock(ch_mult[1], ch_mult[1], ch_mult[0], cond_dim, has_attn=False, context_dim=context_dim)
        
        num_groups_out = min(32, ch_mult[0])
        while ch_mult[0] % num_groups_out != 0:
            num_groups_out //= 2
            
        self.out_norm = nn.GroupNorm(num_groups=num_groups_out, num_channels=ch_mult[0], eps=1e-6)
        self.out_act = nn.SiLU()
        self.conv_out = nn.Conv2d(ch_mult[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor, pooled_context: torch.Tensor) -> torch.Tensor:
        """
        Predicts the velocity vector field v_theta(x_t, t, c) for Flow Matching.
        Args:
            x: (B, 3, H, W) noisy image state at time t
            t: (B,) time scalar in [0, 1]
            context: (B, L, context_dim) token sequence representations
            pooled_context: (B, context_dim) sentence representation
        Returns:
            v: (B, 3, H, W) velocity field
        """
        cond_emb = self.time_embed(t) + self.text_pooled_proj(pooled_context)
        
        h = self.conv_in(x)
        h, skip1 = self.down1(h, cond_emb, context)
        h, skip2 = self.down2(h, cond_emb, context)
        h, skip3 = self.down3(h, cond_emb, context)
        
        h = self.mid_res1(h, cond_emb)
        h = self.mid_attn(h, context)
        h = self.mid_res2(h, cond_emb)
        
        h = self.up3(h, skip3, cond_emb, context)
        h = self.up2(h, skip2, cond_emb, context)
        h = self.up1(h, skip1, cond_emb, context)
        
        out = self.conv_out(self.out_act(self.out_norm(h)))
        return out
