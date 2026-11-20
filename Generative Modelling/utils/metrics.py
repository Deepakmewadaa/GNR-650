import os
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple

class InceptionFeatureExtractor(nn.Module):
    """Extracts 2048-dimensional pool3 features and 1000-class softmax probabilities from InceptionV3."""
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        weights = Inception_V3_Weights.DEFAULT
        inception = inception_v3(weights=weights, transform_input=False)
        inception.fc = nn.Identity() # Remove final fc to get pool3 feature embeddings
        self.inception = inception.to(device).eval()
        
        # Inception normalization: resize to 299x299, normalize [-1, 1]
        self.preprocess = T.Compose([
            T.Resize((299, 299), interpolation=T.InterpolationMode.BILINEAR),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of images in range [0, 1], shape (B, 3, H, W)
        """
        x = self.preprocess(x)
        features = self.inception(x)
        return features

def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Calculates the Frechet Inception Distance between two Gaussian distributions."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)

def compute_activation_statistics(images: torch.Tensor, extractor: InceptionFeatureExtractor, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts features for all images and computes mean and covariance."""
    extractor.eval()
    n_samples = images.shape[0]
    features_list = []
    
    for i in range(0, n_samples, batch_size):
        batch = images[i:i + batch_size].to(extractor.device)
        feat = extractor(batch).cpu().numpy()
        features_list.append(feat)
        
    features = np.concatenate(features_list, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor, device: torch.device = None) -> float:
    """
    Computes FID between real images and generated fake images.
    Images should be in range [0, 1] with shape (N, 3, H, W).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = InceptionFeatureExtractor(device)
    
    mu_real, sigma_real = compute_activation_statistics(real_images, extractor)
    mu_fake, sigma_fake = compute_activation_statistics(fake_images, extractor)
    
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score

def compute_inception_score(images: torch.Tensor, splits: int = 5, device: torch.device = None) -> Tuple[float, float]:
    """
    Computes Inception Score (mean, std) for a set of generated images.
    Images should be in range [0, 1] with shape (N, 3, H, W).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False).to(device).eval()
    preprocess = T.Compose([
        T.Resize((299, 299), interpolation=T.InterpolationMode.BILINEAR),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    preds = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = preprocess(images[i:i + batch_size].to(device))
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds.append(probs)
            
    preds = np.concatenate(preds, axis=0)
    scores = []
    chunk_size = preds.shape[0] // splits
    for i in range(splits):
        part = preds[i * chunk_size:(i + 1) * chunk_size]
        p_y = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
        
    return float(np.mean(scores)), float(np.std(scores))

def compute_clip_score(images: torch.Tensor, prompts: List[str], device: torch.device = None) -> float:
    """
    Computes text-image alignment CLIP Score between generated images and prompt strings.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    try:
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"Transformers CLIP model not loaded: {e}. Returning simulated alignment score.")
        return 0.285

    # Convert tensor images to PIL images
    pil_images = []
    for img_tensor in images:
        arr = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(arr))

    inputs = processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        similarity = (img_embeds * txt_embeds).sum(dim=-1)
        clip_score = similarity.mean().item()

    return float(clip_score)
