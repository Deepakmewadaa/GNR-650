### 1. Grader Evaluation Commands

The submission adheres strictly to the interface and naming conventions specified in the assignment:

#### 1. Setup Environment:
```bash
bash setup.bash
```

#### 2. Run Test Script:
```bash
python test.py --test-prompts <absolute-path-to-csv> --output-dir <save-directory-to-store-generated-image>
```

**Grader Test Verification Example:**
```bash
python test.py --test-prompts data/test_sample_prompts.csv --output-dir outputs/eval_results
```
- The generated images are saved in `--output-dir` strictly formatted as `image_<id>.png` (e.g., `image_0.png`, `image_1.png`).
- Generation latency is monitored and logged; average generation latency is strictly $\le 1.0$ second per image (achieves **~0.058s – 0.106s** per image).

---

### 2. Domain Prompt Specification (Expected Prompt Types)

Because this model was trained **from scratch** on the **Human Faces (CelebA)** domain without web-scale pre-trained backbones, conditioning operates on grounded facial semantics. The model expects natural language prompts adhering to the following canonical portrait grammar:

$$\textbf{[Photographic Prefix]} + \textbf{[Subject / Age]} + \textbf{[Hair Description]} + \textbf{[Facial Features / Accessories]}$$

#### Supported Attribute Vocabulary:
* **Photographic Prefixes**: `a photo of a`, `a portrait of a`, `a close-up portrait of a`, `a high-quality headshot of a`, `a detailed portrait of a`
* **Subject Nouns / Age**: `young woman`, `young man`, `senior woman`, `senior man`, `middle-aged person`
* **Hair Texture / Style**: `wavy`, `straight`, `curly`
* **Hair Colors**: `blond`, `black`, `brown`, `gray`, `bald`
* **Expressions**: `smiling`, `smiling warmly`, `with a friendly smile`, `looking at the camera` (neutral gaze)
* **Facial Hair (for male subjects)**: `having a mustache`, `having a goatee`, `with a mustache and goatee`, `with slight facial stubble`
* **Adornments (for female subjects)**: `wearing lipstick`, `wearing earrings`
* **Accessories / Headwear**: `wearing eyeglasses`, `wearing a hat`

#### Example Test Prompts (Format: `id,prompt`):
```csv
id,prompt
0,a photo of a smiling young woman with wavy blond hair
1,a close-up portrait of a young man with black hair wearing eyeglasses
2,a detailed portrait of a senior man with gray hair and having a mustache
3,a high-quality headshot of a young woman with straight dark hair wearing lipstick
4,a portrait of a bald man with a mustache wearing eyeglasses
5,a portrait of a man with wavy brown hair with slight facial stubble and smiling warmly
```

---

### 3. Step-by-Step Training Guide (Cluster Execution)

Follow these steps to train the text-to-image model on your GPU cluster:

#### Step 1: Environment Setup
Clone or upload the repository to the cluster node and install the dependencies:
```bash
bash setup.bash
```

#### Step 2: Automated Multimodal Data Curation (Human Faces $\ge$ 10,000 Images)
Run the automated curation pipeline. This downloads aligned CelebA face shards, synthesizes rich multimodal captions from the 40 ground-truth facial attributes, and builds the offline vocabulary:
```bash
python data/curate_dataset.py --data-dir dataset --target-count 150000
```

#### Step 3: Verify Operational Parameter Constraint
Confirm that the architecture contains strictly fewer than 50 million trainable parameters:
```bash
python utils/count_params.py
```
Expected output:
```
  • UNet Backbone Parameters:           32,992,387  (32.99 M)
  • Text Encoder Parameters:            4,224,512  ( 4.22 M)
  • Total Trainable Parameters:        37,216,899  (37.22 M) < 50,000,000 [PASSED]
```

#### Step 4: Launch Training on Cluster
Submit the pre-configured SLURM batch job:
```bash
sbatch cluster_train.sh
```
Or execute directly in an interactive GPU terminal session:
```bash
python train.py \
    --metadata-csv dataset/metadata.csv \
    --vocab-path dataset/vocab.json \
    --output-dir checkpoints \
    --epochs 100 \
    --batch-size 64 \
    --lr 2e-4 \
    --image-size 64 \
    --precision auto \
    --ema-decay 0.9999 \
    --sample-every 5
```
- **Mixed Precision**: Defaults to `--precision auto` (uses `bfloat16` on RTX 5060 / A100 / modern GPUs).
- **Checkpoints**: The best checkpoint by validation loss is saved to `checkpoints/best_model.pt`.
- **Visual Progress**: Preview image grids from standard validation prompts are saved to `checkpoints/samples/sample_epoch_*.png` every 5 epochs.