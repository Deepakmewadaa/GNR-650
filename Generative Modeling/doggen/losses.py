import torch
from torch.nn import functional as F


def discriminator_loss(real, fake, mismatch, labels, real_classes, valid_mismatch):
    wrong = F.relu(1 + mismatch)
    wrong = (wrong * valid_mismatch).sum() / valid_mismatch.sum().clamp_min(1)
    return F.relu(1-real).mean() + .5 * (F.relu(1+fake).mean() + wrong) + F.cross_entropy(real_classes, labels)


def generator_loss(fake_score, fake_classes, labels):
    return -fake_score.mean() + F.cross_entropy(fake_classes, labels)


def augment(x):
    # Shared differentiable color/translation augmentation for real and fake.
    b = x.shape[0]
    x = x + (torch.rand(b,1,1,1,device=x.device)-.5)*.2
    x = x * (.8 + torch.rand(b,1,1,1,device=x.device)*.4)
    shifts = torch.randint(-2,3,(2,),device='cpu').tolist()
    return torch.roll(x, shifts, (2,3))
