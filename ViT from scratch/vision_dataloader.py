import os
from PIL import Image
import numpy as np

class VisionDataLoader:
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=32, shuffle=True, 
                 class_to_idx=None, data_fraction=1.0, sampling_strategy='uniform',
                 val_split=0.0, split='all', seed=42):
        """
        Loads an image classification dataset from a directory structure:
        dataset_path/
            class_0/
                img1.png
                img2.png
            class_1/
                img3.png
                ...
        
        Args:
            dataset_path: Root folder of the dataset.
            img_size: Target (H, W) tuple or int.
            batch_size: Batch size for iteration.
            shuffle: Whether to shuffle data per epoch.
            class_to_idx: Optional dict mapping class folder names to integer class IDs.
            data_fraction: Fraction of dataset to use (e.g. 0.1, 0.25, 0.5, 1.0) for data scaling.
            sampling_strategy: 'uniform' (balanced per class) or 'non_uniform' (random slice).
            val_split: Fraction to reserve for validation (e.g. 0.2 for 20% validation).
            split: 'train', 'val', or 'all'.
            seed: Random seed for deterministic train/val splitting.
        """
        self.dataset_path = dataset_path
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.val_split = val_split
        self.split = split
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
            
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        self.classes = sorted(subdirs)
        
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            
        self.image_paths = []
        self.labels = []
        
        # Gather all image paths and labels per class
        per_class_samples = {cls_name: [] for cls_name in self.classes}
        for cls_name in self.classes:
            cls_dir = os.path.join(dataset_path, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(cls_dir, fname)
                    if cls_name in self.class_to_idx:
                        per_class_samples[cls_name].append((full_path, self.class_to_idx[cls_name]))

        # Handle Train / Validation Split per class (stratified)
        if val_split > 0.0 and split in ('train', 'val'):
            rng = np.random.RandomState(seed)
            for cls_name, samples in per_class_samples.items():
                indices = np.arange(len(samples))
                rng.shuffle(indices)
                val_count = max(1, int(len(samples) * val_split))
                val_idx_set = set(indices[:val_count])
                
                if split == 'val':
                    selected = [samples[i] for i in range(len(samples)) if i in val_idx_set]
                else:  # 'train'
                    selected = [samples[i] for i in range(len(samples)) if i not in val_idx_set]
                per_class_samples[cls_name] = selected

        # Handle Data Scaling (Core Component 3)
        if data_fraction < 1.0:
            if sampling_strategy == 'uniform':
                for cls_name, samples in per_class_samples.items():
                    k = max(1, int(len(samples) * data_fraction))
                    selected = samples[:k]
                    for path, label in selected:
                        self.image_paths.append(path)
                        self.labels.append(label)
            else:  # non_uniform sampling
                all_samples = []
                for samples in per_class_samples.values():
                    all_samples.extend(samples)
                k = max(1, int(len(all_samples) * data_fraction))
                for path, label in all_samples[:k]:
                    self.image_paths.append(path)
                    self.labels.append(label)
        else:
            for samples in per_class_samples.values():
                for path, label in samples:
                    self.image_paths.append(path)
                    self.labels.append(label)
                    
        self.num_samples = len(self.image_paths)
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        self.augment = (split == 'train')
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def _load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(self.img_size, Image.BILINEAR)
            
            # Simple data augmentation for training
            if self.augment and np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
            img_array = np.array(img, dtype=np.float32) / 255.0
            # Convert (H, W, C) to (C, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))
            
            # ImageNet normalization (zero-mean, unit-variance)
            img_array = (img_array - self.mean) / self.std
            return img_array
        except Exception as e:
            # Return blank image in case of error
            return np.zeros((3, self.img_size[0], self.img_size[1]), dtype=np.float32)

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
            
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_idx:end_idx]
        
        batch_paths = [self.image_paths[idx] for idx in batch_indices]
        batch_labels = [self.labels[idx] for idx in batch_indices]
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            batch_images = list(executor.map(self._load_image, batch_paths))
            
        self.current_idx = end_idx
        
        return np.array(batch_images), np.array(batch_labels)
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

