# Verified status and remaining work

This is a working baseline, not a claim of converged dog generation or a completed
assignment submission. The data and learned model are real; final quality and
target-GPU compliance must be established with longer training and evaluation.

## Completed

- Read all three pages of the supplied assignment, including its setup/CSV contract.
- Created `.venv` with Python 3.11 and installed pinned dependencies.
- Implemented Windows PowerShell and Linux bash setup; bash setup passed locally
  on macOS. Native Windows and Linux/CUDA execution have not been tested here.
- Downloaded the pinned archive and metadata, verified SHA256, and curated actual
  images. 20,580 candidates -> 20,431 accepted, 149 rejected.
- Dataset splits: 16,345 train, 2,043 validation, 2,043 test, across 120 breeds.
- Independent audit checked every retained PNG, file/pixel hashes, captions,
  metadata and exact/perceptual uniqueness: passed (reports/dataset-audit.json).
- Source manifest SHA256:
  `6e2a66680ba301af83e3b49d1139ce86a587e14acb9fdae732932cd40bd2b429`.
- Implemented the complete randomly initialized text encoder, conditional generator,
  discriminator, adversarial/conditioning losses, EMA and resumable optimizer state.
- Exact deployed parameter count: 1,847,523, including 91,264 text parameters.
  Entire train-time architecture: 3,127,228 parameters.
- Full corpus rerun reproduced the identical manifest SHA256 and split counts
  using cached source files (reports/curation-rerun.log).
- Five tests passed, covering byte tokenization, text influence, gradient flow,
  exact parameter budget, BK-tree equivalence, CSV CLI, duplicate curation,
  deterministic splitting and repair of a deliberately corrupted processed image.
- Actual MPS training reached step 2,510, including a checkpoint resume test,
  using only random-initialized project weights. The default checkpoint is
  checkpoints/latest.pt (format 2). Losses remained finite, but samples are poor;
  this is not a converged dog generator. The global-sum discriminator experiment
  continued the same random-origin run; no external weights were introduced.
- The exact CSV entrypoint generated image_0.png, image_1.png and image_5.png
  from examples/prompts.csv, including an output directory containing a space.
  The system Python entrypoint successfully re-executed in the project venv.
- Batch-one MPS benchmark, 100 warmed images: mean 18.24 ms, p95 20.53 ms,
  maximum 26.09 ms. First cold image was 1.2045 s, exceeding one second on this
  Mac. Loading took 0.0792 s. Timing includes tokenization, forward, transfer and
  PNG encoding; excludes file write and checkpoint load. These are not CUDA
  measurements and do not establish compliance on the evaluation machine.
- FID/IS/CLIP evaluation ran successfully on 20 validation images from the step-500
  revised-model checkpoint. Diagnostic FID=365.3433, IS=1.0352 +/- 0.0197,
  CLIP Score=18.7825. This is explicitly low-sample smoke evidence, not a quality
  claim or final model selection result (reports/metrics-smoke.json).
- All pinned core Windows CPython 3.11 wheels resolved. Official CUDA 13.0
  PyTorch/torchvision wheels resolved for Windows and Linux x86_64.
  This verifies package availability, not native OS/GPU execution.

## Final development-checkpoint diagnostic

At step 2,510, 100 validation images evaluated on CPU produced FID 376.8087,
Inception Score 1.01364 +/- 0.00298 and CLIP Score 19.05345. These values describe
poor development outputs. The 100-image FID is explicitly low-sample and unreliable
as a final estimate. The 20- and 100-image diagnostics use different sample counts
and training states; do not interpret them as a controlled improvement comparison.
No test-set model selection was performed.

Checkpoint SHA256:
`1ad6231efafb9f0276bd663619e77e128e5d66e58421ab77ac6c402b50878534`.
See reports/metrics-diagnostic.json for exact settings, count and provenance.
The discriminator pooling change did not establish useful image quality in this
short continuation. Run longer GPU experiments and inspect samples before treating
the model as usable. The README's default Windows command starts a fresh model
with the final global-sum discriminator; --resume continues the saved project run.

## Evidence locations

`data/curated/summary.json`, `reports/dataset-audit.json`, `reports/parameters.json`,
`reports/tests.txt`, `reports/setup-macos.log`, `reports/environment.json`,
`reports/local-training.log`, `reports/resume-training.log`,
`reports/discriminator-pooling-training.log`, `reports/latency-mps.json`,
`reports/inference-contract.log`, `checkpoints/latest.pt` and
`checkpoints/training.jsonl` contain actual execution evidence. The source archive
and all curated images are present locally. Large data/environment files are
excluded from version control; copy curated data or recreate it on Windows.

## Still required before claiming submission readiness

1. Train to useful visual quality on the Windows GPU. The short development run is
   not evidence of convergence, fidelity or learned semantic prompt alignment.
2. Evaluate whole validation split, select configuration/checkpoint, then freeze it
   and run held-out test FID, IS and CLIP Score. Metric models must stay evaluation-only.
3. Measure the submitted checkpoint on the evaluation-class Linux/CUDA 13.x 8GB
   machine. No RTX 5060, CUDA latency or peak VRAM result is available from this Mac.
   The hardware budget and the <=1-second requirement remain unverified there.
4. Improve caption coverage if arbitrary action/environment prompts are expected.
   Current captions do not label actions or semantic background objects.
5. Write and visually verify the <=8-page CVPR 2027 report using the official author
   kit, citing all sources and AI assistance. This coding-focused delivery supplies
   methodology and evidence; it does not include a finalized CVPR paper.
6. Review source usage terms and course honour-code requirements, package a trained
   checkpoint with the code, test the exact grading commands on Linux and submit
   to the required GitHub/Moodle destinations. No repository was published here.

Never substitute fabricated values for missing metrics. A successful forward pass
or nonzero prompt sensitivity is not a quality/prompt-alignment measurement.
