# Architecture decision and experiment protocol

The hard deployment boundary favors one-pass generation. A small diffusion model
can fit 50M parameters, but multiple denoising evaluations multiply latency and
increase training cost. Distilling diffusion needs a teacher and complicates the
from-scratch requirement. A plain caption-concatenated DCGAN is easy to implement
but can ignore text. We choose a small residual conditional GAN with projection
conditioning and an auxiliary supervised breed loss. It needs no external model
weights. This is a reasoned starting architecture, not a claim that competing
architectures have been empirically beaten.

A UTF-8 tokenizer maps bytes to 256 values plus padding and end symbols, with
192-token truncation. It is fixed, not pretrained and does not inspect held-out
captions to construct a vocabulary. A 64-wide embedding and 128-wide GRU produce
a 128-dimensional text vector. Masked mean pooling combined with the last nonpadding state retains early breed
information and avoids learning from long padding suffixes. All text weights are learned from scratch.

The generator concatenates a 96-dimensional Gaussian noise vector and text,
projects to 256x4x4, then applies four residual upsampling blocks through
128x8x8, 64x16x16, 32x32x32, 32x64x64. Text predicts affine normalization parameters
at both convolutions of every block. Batch normalization, ReLU, nearest-neighbor
upsampling and 3x3 convolutions avoid expensive attention or transposed convolution.
Output tanh maps to [-1,1]. There is no decoder, encoder or hidden retrieval model.

The discriminator downsamples through 32, 64, 128, 256 channels with spectrally
normalized convolutions, global sum pooling, an unconditional scalar and a
normalized text projection inner product. A separate 120-way breed head encourages
semantic structure. Generator and discriminator share the text vector; D updates
detach it, and G updates train it. The optimizer for G includes every text weight.

D loss = mean relu(1-D(real,correct)) + 0.5 mean relu(1+D(fake,correct))
+ 0.5 mean relu(1+D(real,wrong)) + breed cross entropy on real images.
Wrong captions are rolled within a batch; identical captions are masked.
G loss = -mean D(fake,correct) + breed cross entropy on fake images
+ 0.2 cosine alignment loss between projected text and detached real-image features.
No pretrained image encoder participates in this loss. The D projection parameters
are frozen during G updates but retain gradients with respect to their text input.

Both sides receive differentiable brightness/contrast and small translation
augmentation. Discriminator spectral normalization, Adam (lr .0002, betas 0,.9),
G gradient clipping (norm 10), optional CUDA fp16/GradScaler and EMA are included.
EMA decay is .9 for the first 100 steps, .999 afterward. EMA batchnorm buffers track
the live generator. No horizontal flips: they would contradict left/right captions.
A 32-image default batch is intended as an 8GB starting point, not measured target
memory evidence. Reduce batch size if necessary. Checkpoints are written atomically.

64x64 reduces training and batch-one inference costs; the tradeoff is limited fine
breed detail. Dog anatomy, caption coverage and adversarial instability remain real
risks. Starting compact leaves room to increase width after measuring quality.

## Experiments to perform on the Windows GPU

1. Train the base model to 100k optimizer steps, keeping checkpoint evidence and
   evaluating validation metrics at meaningful intervals, not just GAN losses.
2. Compare `--base 16` and `--base 32` using matched steps/seeds. Record exact counts,
   batch-one latency, FID/IS/CLIP and sample grids. These are implemented width options.
3. Inspect same-noise breed/color/location prompt swaps to test text dependence.
   Unit tests establish a gradient path, not learned semantic alignment.
4. If conditioning is weak, improve captions before widening the model. Proposed
   loss ablations require code edits and must be reported as new experiments.
5. Freeze model choices using validation, run held-out test metrics once, and
   measure cold and warm single-image latency on Linux CUDA 13.x with 8GB VRAM.

Do not claim these planned experiments were performed. GPU training can take hours;
there is no evidence-based promise of convergence at a fixed step count.

References: Miyato & Koyama, cGANs with Projection Discriminator,
https://arxiv.org/abs/1802.05637; Miyato et al., Spectral Normalization for GANs,
https://arxiv.org/abs/1802.05957. This code is an original compact PyTorch baseline
inspired by those mechanisms; no pretrained checkpoints or copied training code.

## Observed development failure and correction

The initial short run collapsed to approximately flat scene colors. The optimizer
was updating and discriminator gradients were nonzero, but real/fake score
separation was very weak. A global-mean-pooled discriminator was replaced with
global sum pooling, increasing the spatial discrimination signal without changing
parameter count or deployed generation code. Continuation uses only checkpoints
trained in this project from random weights, never external/pretrained weights.
The pooling experiment is logged separately. This change must be judged from
samples and metrics, not assumed to solve adversarial collapse. New training logs
also record discriminator separation, real breed accuracy and generated spatial
standard deviation to expose this failure earlier.
