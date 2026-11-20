import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from typing import Tuple, Optional
from models.tokenizer import SimpleTokenizer

class MultimodalImageDataset(Dataset):
    """
    Multimodal Dataset for text-conditioned image generation.
    Loads paired (image, caption) records, applies data augmentation,
    normalizes pixels to [-1, 1], and tokenizes text conditioning.
    """
    def __init__(
        self,
        metadata_csv: str,
        split: str = "train",
        image_size: int = 64,
        max_length: int = 64,
        tokenizer: Optional[SimpleTokenizer] = None,
        augment: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.max_length = max_length
        self.tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer()
        
        self.records = []
        if os.path.exists(metadata_csv):
            with open(metadata_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if split is None or row.get("split", "train") == split:
                        # Normalize image path across Windows and Linux
                        raw_path = row.get("image_path", "")
                        # Handle Windows paths on Linux
                        if "\\" in raw_path:
                            raw_path = raw_path.replace("\\", "/")
                        if "dataset/" in raw_path:
                            raw_path = raw_path[raw_path.index("dataset/"):]
                        elif "Images/" in raw_path:
                            raw_path = os.path.join(os.path.dirname(metadata_csv), raw_path[raw_path.index("Images/"):])
                        row["image_path"] = raw_path
                        self.records.append(row)

        if len(self.records) == 0:
            raise RuntimeError(
                f"[!] Dataset is empty for split='{split}' in '{metadata_csv}'.\n"
                f"    Please run the data curation pipeline first:\n"
                f"    python data/curate_dataset.py --target-count 10000"
            )
                        
        transforms_list = []
        if augment and split == "train":
            transforms_list.extend([
                T.Resize((int(image_size * 1.15), int(image_size * 1.15)), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomCrop((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05)
            ])
        else:
            transforms_list.extend([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop((image_size, image_size))
            ])
            
        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # maps [0, 1] -> [-1, 1]
        ])
        self.transform = T.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.records[idx]
        img_path = row["image_path"]
        caption = row.get("caption", "a dog")
        
        # Open and transform image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback if image file is missing or corrupted
            image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
            
        pixel_tensor = self.transform(image)
        
        # Tokenize text caption
        token_ids_list = self.tokenizer.encode(caption, max_length=self.max_length)
        token_ids = torch.tensor(token_ids_list, dtype=torch.long)
        attention_mask = (token_ids != self.tokenizer.pad_id).long()
        
        return pixel_tensor, token_ids, attention_mask

def get_dataloader(
    metadata_csv: str,
    split: str = "train",
    batch_size: int = 32,
    image_size: int = 64,
    num_workers: int = 2,
    shuffle: bool = True
) -> DataLoader:
    dataset = MultimodalImageDataset(
        metadata_csv=metadata_csv,
        split=split,
        image_size=image_size,
        augment=(split == "train")
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )
    return dataloader
