from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import read_manifest


class Dogs(Dataset):
    def __init__(self, manifest, split='train', breeds=None):
        self.root = Path(manifest).parent
        rows = read_manifest(manifest)
        self.breeds = breeds or sorted({r['breed'] for r in rows if r['split'] == 'train'})
        self.rows = [r for r in rows if r['split'] == split]
        if not self.rows:
            raise ValueError(f'Empty {split} split')
        self.labels = {b:i for i,b in enumerate(self.breeds)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['image_path']) as image:
            array = np.array(image.convert('RGB'), dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array).permute(2,0,1) / 127.5 - 1
        return tensor, row['caption'], self.labels[row['breed']]
