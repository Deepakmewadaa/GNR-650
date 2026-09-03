import os
import sys
import argparse
import json
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
    parser = argparse.ArgumentParser(description="GNR650 Vision Transformer Training Pipeline")
    parser.add_argument("--train-path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val-path", type=str, default=None, help="Path to validation dataset (optional)")
    parser.add_argument("--img-size", type=int, default=224, help="Image resolution")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument("--d-model", type=int, default=128, help="Embedding dimension (Core Component 2)")
    parser.add_argument("--d-ff", type=int, default=512, help="FFN dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads (Core Component 2)")
    parser.add_argument("--num-blocks", type=int, default=8, help="Transformer blocks (Core Component 2)")
    parser.add_argument("--num-classes", type=int, default=100, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "inv_sqrt", "constant"], help="Learning rate scheduler")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--tokenizer", type=str, default="fovi", choices=["fovi", "standard"], help="Tokenizer type")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of training data to hold out for validation (e.g. 0.2 for 20%%)")
    parser.add_argument("--data-fraction", type=float, default=1.0, help="Fraction of data for scaling (Core Component 3: 0.1, 0.25, 0.5, 1.0)")
    parser.add_argument("--sampling", type=str, default="uniform", choices=["uniform", "non_uniform"], help="Sampling strategy (Core Component 3)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--exp-name", type=str, default="baseline", help="Experiment name for logging")
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
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join("logs", f"train_{timestamp}_{args.exp_name}.log")
    sys.stdout = DualLogger(log_file_path)

    print("=" * 65)
    print(f" Starting Experiment: {args.exp_name}")
    print(f" Log File:           {log_file_path}")
    print("=" * 65)
    print(f" Architecture: {args.num_blocks} blocks, {args.num_heads} heads, d_model={args.d_model}, d_ff={args.d_ff}")
    print(f" Tokenizer:    {args.tokenizer}")
    print(f" Val Split:    {args.val_split * 100:.1f}%")
    print(f" Data Scale:   {args.data_fraction * 100:.1f}% ({args.sampling} sampling)")
    print(f" Epochs:       {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr} ({args.scheduler} decay)")
    print("-" * 65)

    # 1. Setup DataLoaders
    val_loader = None
    if args.val_path and os.path.exists(args.val_path):
        print(f" Loading training dataset from: {args.train_path}...")
        train_loader = VisionDataLoader(
            dataset_path=args.train_path,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            shuffle=True,
            data_fraction=args.data_fraction,
            sampling_strategy=args.sampling
        )
        print(f" Loading validation dataset from: {args.val_path}...")
        val_loader = VisionDataLoader(
            dataset_path=args.val_path,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            shuffle=False,
            class_to_idx=train_loader.class_to_idx
        )
    else:
        # Automatically split single train directory into Train & Validation
        print(f" Loading dataset from: {args.train_path} (holding out {args.val_split*100:.1f}% for validation)...")
        train_loader = VisionDataLoader(
            dataset_path=args.train_path,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            shuffle=True,
            data_fraction=args.data_fraction,
            sampling_strategy=args.sampling,
            val_split=args.val_split,
            split='train' if args.val_split > 0.0 else 'all'
        )
        if args.val_split > 0.0:
            val_loader = VisionDataLoader(
                dataset_path=args.train_path,
                img_size=(args.img_size, args.img_size),
                batch_size=args.batch_size,
                shuffle=False,
                class_to_idx=train_loader.class_to_idx,
                val_split=args.val_split,
                split='val'
            )

    print(f" Train samples: {train_loader.num_samples} ({len(train_loader)} batches/epoch)")
    print(f" Val samples:   {val_loader.num_samples if val_loader else 0}")
    print("-" * 65)

    # 2. Initialize Optimizer & Model
    total_steps = args.epochs * len(train_loader)
    optimizer = Adam(
        lr=args.lr, 
        warmup_steps=args.warmup_steps, 
        total_steps=total_steps, 
        scheduler=args.scheduler, 
        weight_decay=args.weight_decay, 
        d_model=args.d_model
    )
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
        dropout_rate=args.dropout,
        label_smoothing=args.label_smoothing
    )
    model.class_to_idx = train_loader.class_to_idx

    num_params = model.calculate_trainable_parameters()
    macs, flops = model.calculate_macs_and_flops()
    print(f" Trainable Parameters: {num_params:,}")
    print(f" Theoretical MACs:     {macs:,} ({macs/1e6:.2f} M)")
    print(f" Theoretical FLOPs:    {flops:,} ({flops/1e6:.2f} M)")
    print("-" * 65)


    # 3. Train Model
    checkpoint_file = os.path.join(args.checkpoint_dir, f"{args.exp_name}_best.npz")
    default_best_file = os.path.join(args.checkpoint_dir, "best_model.npz")

    start_train = time.time()
    history = model.fit_generator(
        num_epochs=args.epochs,
        dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoint_path=checkpoint_file
    )
    total_train_time = time.time() - start_train

    # Ensure default best_model.npz exists
    if not os.path.exists(default_best_file) and os.path.exists(checkpoint_file):
        import shutil
        shutil.copyfile(checkpoint_file, default_best_file)
    elif not os.path.exists(default_best_file):
        model.save_checkpoint(default_best_file)

    # 4. Save Experiment Log
    log_data = {
        'exp_name': args.exp_name,
        'config': vars(args),
        'num_trainable_params': num_params,
        'macs': macs,
        'flops': flops,
        'total_train_time_sec': total_train_time,
        'history': history
    }
    log_file = os.path.join(args.checkpoint_dir, f"{args.exp_name}_log.json")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)

    print("\n" + "=" * 65)
    print(" TRAINING COMPLETE!")
    print(f" Total Training Time: {total_train_time:.2f} seconds ({total_train_time/60:.2f} minutes)")
    print(f" Checkpoint saved to: {checkpoint_file}")
    print(f" Experiment logs:     {log_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
