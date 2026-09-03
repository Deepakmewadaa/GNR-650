try:
    import cupy as cp
except ImportError:
    import numpy as cp
from encoder_block import EncoderBlock


class VisionTransformerEncoder():
    """
    Plain stack of the existing EncoderBlock. No masking is used - every
    patch (and the CLS token) attends to every other patch, since there is
    no padding or causality concept for a fixed-size image grid.
    """

    def __init__(self, optimizer, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type):
        self.encoder_layers = [
            EncoderBlock(d_model, d_ff, optimizer, num_attention_heads, dropout_rate, data_type)
            for _ in range(block_num)
        ]

    def forward(self, x, training=True):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward(x, mask=None, training=training)
        return x

    def backward(self, grad):
        for encoder_layer in reversed(self.encoder_layers):
            grad = encoder_layer.backward(grad)
        return grad

    def update_weights(self):
        for encoder_layer in reversed(self.encoder_layers):
            encoder_layer.update_weights()
