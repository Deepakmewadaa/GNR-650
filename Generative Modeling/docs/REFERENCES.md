# Sources and assistance

- GNR650 Assignment-3.pdf, supplied in this repository: specification and evaluation contract.
- Khosla et al. (2011), Novel Dataset for Fine-Grained Image Categorization,
  https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf
- Stanford Dogs source: http://vision.stanford.edu/aditya86/ImageNetDogs/
- Alanox Stanford Dogs image/annotation mirror (pinned revision in code):
  https://huggingface.co/datasets/Alanox/stanford-dogs
- Deng et al. (2009), ImageNet, https://www.image-net.org/
- Miyato & Koyama (2018), cGANs with Projection Discriminator,
  https://arxiv.org/abs/1802.05637
- Miyato et al. (2018), Spectral Normalization for Generative Adversarial Networks,
  https://arxiv.org/abs/1802.05957
- Official projection GAN implementation consulted for architectural context:
  https://github.com/pfnet-research/sngan_projection
- PyTorch installation and binary wheels:
  https://pytorch.org/get-started/locally/ and https://download.pytorch.org/whl/cu130/
- TorchMetrics FID documentation:
  https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance
- TorchMetrics Inception Score documentation:
  https://lightning.ai/docs/torchmetrics/stable/image/inception_score
- TorchMetrics CLIP Score documentation:
  https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score
- Evaluation-only CLIP model: https://huggingface.co/openai/clip-vit-base-patch32
- OpenAI Codex (gpt-6.1-sol, T3 Code harness) assisted architecture research,
  implementation, data pipeline execution, tests and documentation. Human authors
  should review generated code and measured evidence and disclose assistance in
  the final assignment report according to the course honour code.

No pretrained weights were used for curation, generation, discriminator, text
encoder or training losses. Only optional evaluation metrics load externally
trained image/text feature extractors. The byte vocabulary is fixed, untrained;
all embeddings and GRU weights are random initialized and trained in this project.
