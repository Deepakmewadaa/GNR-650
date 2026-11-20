import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.unet import TextConditionedUNet
from models.text_encoder import TextEncoder
from models.flow_matching import FlowMatching

def verify_parameters(verbose: bool = True):
    unet = TextConditionedUNet()
    text_encoder = TextEncoder()
    model = FlowMatching(unet, text_encoder)
    
    unet_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    text_encoder_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_all = sum(p.numel() for p in model.parameters())
    
    max_allowed = 50_000_000
    
    if verbose:
        print("=" * 65)
        print("          MODEL PARAMETER CONSTRAINT VERIFICATION (< 50M)     ")
        print("=" * 65)
        print(f"  • UNet Backbone Parameters:         {unet_params:>12,}  ({unet_params / 1e6:>5.2f} M)")
        print(f"  • Text Encoder Parameters:         {text_encoder_params:>12,}  ({text_encoder_params / 1e6:>5.2f} M)")
        print("-" * 65)
        print(f"  • Total Trainable Parameters:      {total_trainable:>12,}  ({total_trainable / 1e6:>5.2f} M)")
        print(f"  • Total Architecture Parameters:   {total_all:>12,}  ({total_all / 1e6:>5.2f} M)")
        print(f"  • Hard Ceiling Limit:              {max_allowed:>12,}  (50.00 M)")
        print(f"  • Margin to Limit:                 {max_allowed - total_trainable:>12,}  ({(max_allowed - total_trainable) / 1e6:>5.2f} M safety buffer)")
        print("=" * 65)
        
    assert total_trainable < max_allowed, (
        f"FAILED: Model parameter constraint violated! Total params {total_trainable:,} >= 50,000,000"
    )
    if verbose:
        print("  [PASSED] Model strictly adheres to the < 50M parameter constraint!\n")
        
    return total_trainable

if __name__ == "__main__":
    verify_parameters(verbose=True)
