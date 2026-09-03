import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR)

try:
    import cupy as cp
except ImportError:
    import numpy as cp

import os
os.environ['FOVI_SAVE_DIR'] = './'
os.environ['FOVI_DATASETS_DIR'] = './'

from optimizer import Adam
from vision_transformer import VisionTransformer
from vision_dataloader import VisionDataLoader

if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 100
    d_model = 64
    d_ff = 128
    num_attention_heads = 4
    block_num = 4
    dropout_rate = 0.1

    optimizer = Adam(lr=1e-3, d_model=d_model, warmup_steps=100)
    model = VisionTransformer(
        optimizer=optimizer,
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=d_model,
        d_ff=d_ff,
        num_attention_heads=num_attention_heads,
        block_num=block_num,
        dropout_rate=dropout_rate,
    )

    dataset_path = r"C:\Users\deepa\Downloads\train\train"
    print(f"Loading real dataset from {dataset_path}...")

    # 1. Initialize the DataLoader pointing to the real dataset directory
    train_loader = VisionDataLoader(
        dataset_path=dataset_path, 
        img_size=(img_size, img_size),
        batch_size=8, 
        shuffle=True
    )

    print("Starting training using fit_generator()...")
    # 2. Train the model using the new fit_generator method
    model.fit_generator(
        num_epochs=5, 
        dataloader=train_loader
    )

    print("Training pipeline finished successfully!")
