try:
    import cupy as cp
except ImportError:
    import numpy as cp
from dropout import Dropout
from layer_normalization import LayerNormalization
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward
class EncoderBlock():
    def __init__(self, d_model, d_ff, optimizer, num_attention_heads, dropout_rate, data_type):
        self.d_model = d_model
        self.d_ff = d_ff
        self.optimizer = optimizer
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.data_type = data_type
        self.dropout1 = Dropout(self.dropout_rate, self.data_type)
        self.dropout2 = Dropout(self.dropout_rate, self.data_type)
        self.layernorm1 = LayerNormalization(self.optimizer, normalized_shape=self.d_model, eps=1e-5, data_type=self.data_type)
        self.layernorm2 = LayerNormalization(self.optimizer, normalized_shape=self.d_model, eps=1e-5, data_type=self.data_type)
        self.multi_head_attention = MultiHeadAttention(self.optimizer, self.d_model, self.num_attention_heads, self.dropout_rate, self.data_type)
        self.ffn = PositionWiseFeedForward(self.optimizer, self.d_model, self.d_ff, self.dropout_rate, self.data_type)

    def forward(self, x, mask=None, training=True):
        # 1. Pre-LN Self-Attention with Skip Connection
        norm_x1 = self.layernorm1.forward(x)
        attn_out = self.multi_head_attention.forward(norm_x1, norm_x1, norm_x1, mask, training)
        attn_out = self.dropout1.forward(attn_out, training)
        x1 = x + attn_out
        
        # 2. Pre-LN Feed-Forward Network with Skip Connection
        norm_x2 = self.layernorm2.forward(x1)
        ffn_out = self.ffn.forward(norm_x2, training)
        ffn_out = self.dropout2.forward(ffn_out, training)
        out = x1 + ffn_out
        return out

    def backward(self, grad):
        # Gradient flow through FFN branch + residual skip connection
        grad_ffn = self.dropout2.backward(grad)
        grad_ffn = self.ffn.backward(grad_ffn)
        grad_norm2 = self.layernorm2.backward(grad_ffn)
        grad_x1 = grad + grad_norm2

        # Gradient flow through MHA branch + residual skip connection
        grad_attn = self.dropout1.backward(grad_x1)
        grad_q, grad_k, grad_v = self.multi_head_attention.backward(grad_attn)
        grad_attn_in = grad_q + grad_k + grad_v
        grad_norm1 = self.layernorm1.backward(grad_attn_in)
        grad_x = grad_x1 + grad_norm1
        return grad_x

    def update_weights(self):
        self.layernorm2.update_weights()
        self.ffn.update_weights()
        self.layernorm1.update_weights()
        self.multi_head_attention.update_weights()

