 ### What Was Included in .gitignore & Why

  #### 1. Datasets & Raw Archives (GitHub Size Limits & AGENTS.md Compliance)

  • What is ignored: data/raw/, data/curated/images/, *.tar, *.tar.gz, *.tgz, *.zip, *.7z.
  • Why: Section 3.1 of Assignment-3.pdf requires a curated dataset of ≥ 10, 000 images (currently > 20, 000 images in data/curated/images/). The raw Stanford Dogs archive
  (images.tar.gz) is ~740 MB. GitHub enforces a strict 100 MB hard limit per file and blocks pushes exceeding it. The dataset curation pipeline is deterministic and re-runnable
  locally via python -m doggen.curate. Curated image metadata (manifest.jsonl and summary.json) remains tracked.

  #### 2. Model Checkpoints & Submission Contract

  • What is ignored: checkpoints/*.pt, checkpoints/*.ckpt, checkpoints/*.pth, checkpoints/*.safetensors, etc.
  • What is safelisted: !checkpoints/latest.pt
  • Why: Section 1.1 and Section 5 of Assignment-3.pdf specify the grading contract:
    bash setup.bash
    python test.py --test-prompts <absolute-path-to-csv> --output-dir <save-directory-to-store-generated-image>
  test.py:20 defaults to loading checkpoints/latest.pt. The checkpoint file is ~45.1 MB, safely within GitHub's 100 MB limit, so it must be committed for the evaluation harness
  to function. However, as 3 teammates run training runs on GPU, intermediate weights (step_*.pt, diagnostic.pt, etc.) will each be ~45 MB. Tracking multiple intermediate runs
  would quickly exceed GitHub's 2 GB recommended repository limit. Safelisting only latest.pt prevents repository bloat.
  #### 3. Cross-Platform OS Noise (macOS / Windows / Linux)

  • What is ignored:
      • macOS: .DS_Store, ._*, .Spotlight-V100, .Trashes
      • Windows: Thumbs.db, Desktop.ini, ehthumbs.db, $RECYCLE.BIN/
      • Linux: *~, .fuse_hidden*, .directory
  • Why: The project documentation details Windows training (PowerShell CUDA), Linux evaluation, and local macOS development. Without these rules, .DS_Store and Windows thumbnail
  files will pollute commits and cause constant merge conflicts.

  #### 4. CVPR 2027 LaTeX Report Auxiliary Files
  • What is ignored: *.aux, *.bbl, *.blg, *.fdb_latexmk, *.fls, *.synctex.gz, *.toc, *.out, etc.
  • Why: Section 1.1 of Assignment-3.pdf requires a technical report formatted using the official CVPR 2027 LaTeX Template. Compiling LaTeX generates dozens of auxiliary files
  that create messy git diffs between collaborators. Source files (.tex, .bib, styles, and figures) remain tracked.

  #### 5. Virtual Environments & Platform-Specific Binaries

  • What is ignored: .venv/, venv/, env/, ENV/, .env/
  • Why: Virtual environments contain platform-specific compiled binaries and machine-specific absolute paths. Per AGENTS.md and README.md, each team member should create their
  local .venv using setup.bash (Linux/macOS) or setup.ps1 (Windows).

  #### 6. Inference Outputs & Temporary Files

  • What is ignored: outputs/, generated/, runs/, wandb/, *.tmp, *.temp, *.bak
  • Why: test.py saves output images (image_<id>.png) to --output-dir. Atomic checkpoint writing in train.py:20 uses .tmp files. Generated test images and temporary artifacts
  should never be checked into git.

  #### 7. IDE, Editor & Notebook Artifacts

  • What is ignored: .vscode/, .idea/, *.swp, *.swo, .ipynb_checkpoints/
  • Why: Prevents team members from clobbering each other's editor settings, interpreter paths, and Jupyter notebook autosaves.

  #### 8. Audit Logs vs. Ephemeral Logs

  • Rule: General *.log files are ignored, but audit and benchmark logs under reports/ are preserved (!reports/*.log, !reports/**/*.log) because they serve as reproducible
  verification evidence noted in STATUS.md.
  ──────
  ### File Created

  The file has been saved to:

  • .gitignore

  When you are ready to initialize your repository manually, you can run your standard commands (git init, git add ., etc.) and git will cleanly respect these rules without
  committing any OS artifacts, massive dataset archives, or platform-specific binaries.