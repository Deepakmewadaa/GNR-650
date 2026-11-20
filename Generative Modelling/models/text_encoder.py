import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    """
    Trainable Transformer-based Text Encoder trained from scratch.
    Maps tokenized text sequences to contextual token embeddings and a pooled sentence embedding.
    No pre-trained weights or checkpoints are used.
    """
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        max_length: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, embed_dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            input_ids: (Batch_Size, Sequence_Length) integer tensor of token IDs.
            attention_mask: (Batch_Size, Sequence_Length) binary tensor (1 for token, 0 for pad).
        Returns:
            token_embeddings: (Batch_Size, Sequence_Length, embed_dim)
            pooled_embedding: (Batch_Size, embed_dim)
        """
        B, L = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :L, :]
        x = self.drop(x)
        
        pad_mask = None
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)
            
        hidden = self.transformer(x, src_key_padding_mask=pad_mask)
        hidden = self.final_norm(hidden)
        
        # Mean pooling over non-padded tokens for global sentence embedding
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)
            
        return hidden, pooled
