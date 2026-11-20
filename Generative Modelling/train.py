import os
import sys
import copy
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.utils as vutils
from tqdm import tqdm

from models.tokenizer import SimpleTokenizer
from models.text_encoder import TextEncoder
from models.unet import TextConditionedUNet
from models.flow_matching import FlowMatching
from utils.dataset import get_dataloader
from utils.count_params import verify_parameters

class ModelEMA:
    """Maintains an exponential moving average of model parameters for superior generative fidelity."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_amp_dtype(precision_arg: str, device: torch.device):
    if device.type != "cuda":
        return torch.float32, False
    if precision_arg == "bf16" or (precision_arg == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, True
    elif precision_arg == "fp16":
        return torch.float16, True
    return torch.float32, False

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_amp = get_amp_dtype(args.precision, device)
    print(f"Training on device: {device} | Mixed precision: {amp_dtype if use_amp else 'FP32'}")
    print(f"Target image resolution: {args.image_size}x{args.image_size}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # 1. Initialize Tokenizer & Models
    vocab_path = args.vocab_path if os.path.exists(args.vocab_path) else None
    tokenizer = SimpleTokenizer(vocab_file=vocab_path)
    
    text_encoder = TextEncoder(vocab_size=max(tokenizer.vocab_size, 4096), embed_dim=256)
    unet = TextConditionedUNet(in_channels=3, out_channels=3, ch_mult=[64, 128, 256, 384], context_dim=256)
    flow_model = FlowMatching(unet=unet, text_encoder=text_encoder).to(device)

    # Strictly verify the model parameter constraint (< 50M)
    trainable_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"Total Trainable Architecture Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    assert trainable_params < 50_000_000, "Constraint Violated: Trainable parameters must strictly be < 50M!"
    
    # Initialize EMA shadow model
    ema = ModelEMA(flow_model, decay=args.ema_decay)

    # 2. Ensure dataset exists and is non-empty; auto-curate if needed
    if not os.path.exists(args.metadata_csv) or os.path.getsize(args.metadata_csv) < 100:
        print(f"[!] Notice: '{args.metadata_csv}' not found or empty.")
        print("    Automatically launching data curation pipeline (>= 10,000 images)...")
        from data.curate_dataset import curate_dataset
        data_parent = os.path.dirname(args.metadata_csv) or "dataset"
        curate_dataset(data_dir=data_parent, target_count=10000)

    # 3. Multimodal Data Loaders
    workers = args.num_workers
    if os.name == "nt" and workers > 0:
        workers = 0 # Safe multiprocessing on Windows
        
    print(f"Loading dataset from: {args.metadata_csv} (num_workers: {workers})")
    train_loader = get_dataloader(
        metadata_csv=args.metadata_csv,
        split="train",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=workers,
        shuffle=True
    )
    val_loader = get_dataloader(
        metadata_csv=args.metadata_csv,
        split="val",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=workers,
        shuffle=False
    )
    print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    # 3. Optimizer & Scheduler
    optimizer = AdamW(flow_model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)), eta_min=1e-6)

    # Scaler for fp16 if explicitly selected, else standard unscaled for bf16/fp32
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    start_epoch = 0
    best_val_loss = float("inf")

    # Optional Resume
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        flow_model.load_state_dict(checkpoint["model_state_dict"])
        if "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # Test prompts for validation image progression tracking (Human Faces)
    val_test_prompts = [
        "a photo of a smiling young woman with wavy blond hair",
        "a close-up portrait of a young man with black hair wearing eyeglasses",
        "a portrait of a smiling man with brown hair and mustache",
        "a beautiful young woman with dark hair wearing lipstick"
    ]
    val_prompt_tokens = torch.tensor(
        [tokenizer.encode(p, max_length=64) for p in val_test_prompts],
        dtype=torch.long,
        device=device
    )

    # 4. Training Loop
    global_step = 0
    print("\nStarting Flow Matching training from scratch...")
    for epoch in range(start_epoch, args.epochs):
        flow_model.train()
        train_loss_accum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for images, token_ids, attn_mask in pbar:
            images = images.to(device)
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    loss = flow_model.compute_loss(
                        x_1=images,
                        token_ids=token_ids,
                        attn_mask=attn_mask,
                        uncond_prob=0.1 # Classifier-Free Guidance dropout
                    )
            else:
                loss = flow_model.compute_loss(images, token_ids, attn_mask, uncond_prob=0.1)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Update Exponential Moving Average weights
            ema.update(flow_model)

            train_loss_accum += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

        avg_train_loss = train_loss_accum / max(1, len(train_loader))

        # Validation Step
        flow_model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for images, token_ids, attn_mask in val_loader:
                images = images.to(device)
                token_ids = token_ids.to(device)
                attn_mask = attn_mask.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        v_loss = flow_model.compute_loss(images, token_ids, attn_mask, uncond_prob=0.0)
                else:
                    v_loss = flow_model.compute_loss(images, token_ids, attn_mask, uncond_prob=0.0)
                val_loss_accum += v_loss.item()
                
        avg_val_loss = val_loss_accum / max(1, len(val_loader)) if len(val_loader) > 0 else avg_train_loss
        print(f"Epoch {epoch+1} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Generate sample visual previews using EMA model
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            print(f"Generating preview samples using EMA model for epoch {epoch+1}...")
            with torch.inference_mode():
                sample_imgs = ema.shadow.sample(
                    token_ids=val_prompt_tokens,
                    num_steps=12,
                    cfg_scale=4.5,
                    image_size=args.image_size
                )
            grid_path = os.path.join(samples_dir, f"sample_epoch_{epoch+1:03d}.png")
            vutils.save_image(sample_imgs, grid_path, nrow=2, normalize=False)
            print(f"Saved visual sample grid to {grid_path}")

        # Save Checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": flow_model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "vocab": tokenizer.word2idx,
            "config": {
                "image_size": args.image_size,
                "ch_mult": [64, 128, 256, 384],
                "context_dim": 256,
                "vocab_size": tokenizer.vocab_size
            }
        }
        latest_path = os.path.join(args.output_dir, "latest_model.pt")
        torch.save(checkpoint_data, latest_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(checkpoint_data, best_path)
            print(f"[*] New best validation loss ({best_val_loss:.4f})! Saved model to {best_path}")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flow Matching Generative Model from Scratch")
    parser.add_argument("--metadata-csv", type=str, default="dataset/metadata.csv", help="Path to curated dataset metadata CSV")
    parser.add_argument("--vocab-path", type=str, default="dataset/vocab.json", help="Path to vocabulary JSON file")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints and samples")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--image-size", type=int, default=64, help="Image resolution (64)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker threads")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="Exponential Moving Average decay factor")
    parser.add_argument("--sample-every", type=int, default=5, help="Generate sample grid every N epochs")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Mixed precision mode")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    train(args)
