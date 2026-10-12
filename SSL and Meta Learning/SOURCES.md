# Sources

Everything below was implemented from scratch in plain PyTorch. The papers and docs were used for the method
definitions and hyper-parameter defaults only. No code was copied from the listed repositories, and no
pretrained weights or external data were used.

## Self-supervised learning
- T. Chen, S. Kornblith, M. Norouzi, G. Hinton. *A Simple Framework for Contrastive Learning of Visual
  Representations (SimCLR).* ICML 2020. arXiv:2002.05709.
  - Used for: NT-Xent loss, projection head, augmentation recipe, LARS + warmup/cosine schedule.
- M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, A. Joulin. *Emerging Properties in
  Self-Supervised Vision Transformers (DINO).* ICCV 2021. arXiv:2104.14294.
  - Used for: self-distillation, centering/sharpening, multi-crop, EMA teacher, the ResNet-50 recipe.
- J.-B. Grill et al. *Bootstrap Your Own Latent (BYOL).* NeurIPS 2020. arXiv:2006.07733.
  - Used for: asymmetric blur/solarisation probabilities.
- Y. You, I. Gitman, B. Ginsburg. *Large Batch Training of Convolutional Networks (LARS).* arXiv:1708.03888.
- I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine schedule), ICLR 2017;
  *Decoupled Weight Decay Regularization (AdamW)*, ICLR 2019.
- P. Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv:1706.02677.
  - Used for: linear LR scaling and zero-init of the last BN in each residual block.
- X. Chen, K. He. *Exploring Simple Siamese Representation Learning (SimSiam).* CVPR 2021.
  - Used for: the std of L2-normalised features as a collapse diagnostic.
- Q. Garrido, R. Balestriero, L. Najman, Y. LeCun. *RankMe: Assessing the Downstream Performance of Pretrained
  Self-Supervised Representations by Their Rank.* ICML 2023.
  - Used for: the effective-rank diagnostic.
- Z. Wu, Y. Xiong, S. Yu, D. Lin. *Unsupervised Feature Learning via Non-Parametric Instance Discrimination.*
  CVPR 2018.
  - Used for: the weighted kNN monitor.

## Architecture
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition (ResNet).* CVPR 2016.
- S. Ioffe, C. Szegedy. *Batch Normalization.* ICML 2015.
- K. He et al. *Delving Deep into Rectifiers* (Kaiming initialisation). ICCV 2015.

## Learning with noisy labels
- C. Szegedy et al. *Rethinking the Inception Architecture* (label smoothing). CVPR 2016.
- Y. Wang, X. Ma, Z. Chen, Y. Luo, J. Yi, J. Bailey. *Symmetric Cross Entropy for Robust Learning with Noisy
  Labels.* ICCV 2019.
- Z. Zhang, M. R. Sabuncu. *Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy
  Labels.* NeurIPS 2018.
- S. Reed et al. *Training Deep Neural Networks on Noisy Labels with Bootstrapping.* ICLR Workshop 2015.
- C. Northcutt, L. Jiang, I. Chuang. *Confident Learning: Estimating Uncertainty in Dataset Labels.* JAIR 2021.
  - Used for: the confident joint and the noise-rate estimate. The `cleanlab` library itself is NOT used.
- B. Han et al. *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels.* NeurIPS 2018.
  - Used for: the small-loss selection criterion.
- D. Bahri, H. Jiang, M. Gupta. *Deep k-NN for Noisy Labels.* ICML 2020.
  - Used for: kNN label-consensus filtering.

## Out-of-distribution / unknown-class detection
- D. Hendrycks, K. Gimpel. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural
  Networks* (MSP). ICLR 2017.
- D. Hendrycks et al. *Scaling Out-of-Distribution Detection for Real-World Settings* (max-logit). ICML 2022.
- W. Liu, X. Wang, J. Owens, Y. Li. *Energy-based Out-of-distribution Detection.* NeurIPS 2020.
- K. Lee, K. Lee, H. Lee, J. Shin. *A Simple Unified Framework for Detecting Out-of-Distribution Samples and
  Adversarial Attacks* (Mahalanobis). NeurIPS 2018.
- O. Ledoit, M. Wolf. *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.* J. Multivariate
  Analysis 2004.
  - Used for: the idea of covariance shrinkage. Our shrinkage coefficient is fixed, not the LW formula.
- Y. Sun, Y. Ming, X. Zhu, Y. Li. *Out-of-Distribution Detection with Deep Nearest Neighbors* (kNN). ICML 2022.
- W. J. Scheirer et al. *Toward Open Set Recognition.* TPAMI 2013 (open-set problem formulation).
- Metrics: AUROC (Mann–Whitney U), average precision, FPR at 95 % TPR — standard definitions, implemented in
  numpy.

## Documentation and tools
- PyTorch documentation (torch.nn, torch.nn.functional `grid_sample` / `affine_grid` / `conv2d`, AMP, DataLoader,
  `torch.load(weights_only=...)`): https://pytorch.org/docs/stable/
- PyTorch installation pages for CUDA 13 wheels: https://pytorch.org/get-started/locally/ and
  https://pytorch.org/get-started/previous-versions/. Wheel index checked:
  https://download.pytorch.org/whl/cu130/torch/
- Pillow documentation (image modes, `ImageOps.exif_transpose`, `alpha_composite`):
  https://pillow.readthedocs.io/
- NumPy, PyYAML, Matplotlib documentation; PyPI package pages (to check wheel availability for the pinned
  versions).
- pytest (tests). scikit-learn is used only as an optional cross-check of our AUROC/AP implementation in tests.

## AI assistance
- **Claude Code (Anthropic), model Claude Opus 5.5.** It was used to write the codebase, tests and
  documentation: the design, implementation, synthetic-data verification and the report generator, from the
  assignment PDF and our instructions. All design decisions are listed in `docs/decisions.md`. The students
  reviewed the code, ran all real training, and are responsible for the submission.
