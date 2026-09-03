import os
import sys
import argparse
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _sub in ("BaseLayers", "CombinedLayers", "Encoder"):
    _path = os.path.join(_THIS_DIR, _sub)
    if _path not in sys.path:
        sys.path.append(_path)

try:
    import cupy as cp
except ImportError:
    import numpy as cp

from optimizer import Adam
from vision_transformer import VisionTransformer
from vision_dataloader import VisionDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="GNR650 Vision Transformer Evaluation Script")
    parser.add_argument("--test-path", type=str, required=True, help="Absolute path to the test dataset directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_best.npz", help="Path to trained model checkpoint (.npz)")
    parser.add_argument("--img-size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument("--d-model", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--d-ff", type=int, default=512, help="FFN dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-blocks", type=int, default=8, help="Number of transformer encoder blocks")
    parser.add_argument("--num-classes", type=int, default=100, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--tokenizer", type=str, default="fovi", choices=["fovi", "standard"], help="Tokenizer type ('fovi' or 'standard')")
    return parser.parse_args()


from datetime import datetime


class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if '\r' not in message:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()


def main():
    args = parse_args()
    import json
    import numpy as np

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join("logs", f"test_{timestamp}.log")
    sys.stdout = DualLogger(log_file_path)

    # Auto-fallback if default checkpoint not found
    if not os.path.exists(args.checkpoint) and args.checkpoint == "checkpoints/best_model.npz":
        if os.path.exists("checkpoints/baseline_best.npz"):
            args.checkpoint = "checkpoints/baseline_best.npz"

    # Auto-load architecture configuration from checkpoint if present
    if os.path.exists(args.checkpoint):
        try:
            data = np.load(args.checkpoint, allow_pickle=True)
            if 'config_json' in data:
                cfg = json.loads(str(data['config_json'][0]))
                args.img_size = cfg.get('img_size', args.img_size)
                args.patch_size = cfg.get('patch_size', args.patch_size)
                args.d_model = cfg.get('d_model', args.d_model)
                args.d_ff = cfg.get('d_ff', args.d_ff)
                args.num_heads = cfg.get('num_attention_heads', args.num_heads)
                args.num_blocks = cfg.get('block_num', args.num_blocks)
                args.num_classes = cfg.get('num_classes', args.num_classes)
                args.tokenizer = cfg.get('tokenizer_type', args.tokenizer)
        except Exception:
            pass

    print("=" * 65)
    print(" GNR650: Programming Assignment - Vision Transformer Evaluation")
    print("=" * 65)
    print(f" Dataset Path:      {args.test_path}")
    print(f" Checkpoint:        {args.checkpoint}")
    print(f" Architecture:      {args.num_blocks} blocks, {args.num_heads} heads, d_model={args.d_model}, d_ff={args.d_ff}")
    print(f" Tokenizer:         {args.tokenizer}")
    print(f" Image Resolution:  {args.img_size}x{args.img_size} (patch {args.patch_size})")
    print(f" Batch Size:        {args.batch_size}")
    print("-" * 65)

    # 1. Initialize Optimizer & Model
    optimizer = Adam(lr=1e-3, d_model=args.d_model)
    model = VisionTransformer(
        optimizer=optimizer,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=args.num_classes,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_attention_heads=args.num_heads,
        block_num=args.num_blocks,
        tokenizer_type=args.tokenizer,
        dropout_rate=0.0
    )

    # 2. Load Checkpoint if available
    if os.path.exists(args.checkpoint):
        model.load_checkpoint(args.checkpoint)
    else:
        print(f" [WARNING] Checkpoint file '{args.checkpoint}' not found.")
        print(" Running inference with initialized weights for demonstration / benchmark.")

    # 3. Calculate Model Complexity Metrics (Parameters, FLOPs, MACs)
    num_trainable_params = model.calculate_trainable_parameters()
    total_macs, total_flops = model.calculate_macs_and_flops()

    # 4. Load Dataset
    print(f" Loading test dataset from: {args.test_path}...")
    test_loader = VisionDataLoader(
        dataset_path=args.test_path,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
        class_to_idx=model.class_to_idx
    )
    print(f" Found {test_loader.num_samples} test images across {len(test_loader.classes)} classes.")

    # 5. Perform Inference and Measurement
    print(" Running test set inference...")
    eval_results = model.evaluate(test_loader, desc="Testing Evaluation")

    test_accuracy = eval_results['accuracy']
    total_inference_time = eval_results['total_time']
    inference_throughput = eval_results['throughput']

    # 6. Format and Print Required Evaluation Metrics
    def format_number(n):
        if n >= 1e9:
            return f"{n / 1e9:.3f} G"
        elif n >= 1e6:
            return f"{n / 1e6:.3f} M"
        elif n >= 1e3:
            return f"{n / 1e3:.3f} K"
        return f"{n:,}"

    print("\n" + "=" * 65)
    print(" FINAL EVALUATION RESULTS")
    print("=" * 65)
    print(f" - Test-set accuracy:          {test_accuracy:.2f}%")
    print(f" - Total inference time:       {total_inference_time:.4f} seconds")
    print(f" - Inference throughput:       {inference_throughput:.2f} images/second")
    print(f" - MACs:                       {format_number(total_macs)} ({total_macs:,} operations)")
    print(f" - FLOPs:                      {format_number(total_flops)} ({total_flops:,} operations)")
    print(f" - Number of trainable parameters: {num_trainable_params:,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
