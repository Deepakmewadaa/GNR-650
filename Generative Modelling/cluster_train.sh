#!/usr/bin/env bash
#SBATCH --job-name=gnr650_flow_face
#SBATCH --account=itl_proj1
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==============================================================================
# Cluster Training Script: Continuous Flow Matching for Text-to-Image Generation
# Model: < 37.2M Trainable Parameters | Domain: Human Faces (CelebA)
# ==============================================================================
mkdir -p logs checkpoints
set -e

echo "=== Job Started on $(hostname) at $(date) ==="
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
python3 -c "import torch; print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Step 1: Verify environment and parameter constraint (< 50M)
# python3 utils/count_params.py

# Step 2: Automated Data Curation (>= 10,000 images if not already prepared)
# if [ ! -s "dataset/metadata.csv" ] || [ $(wc -l < "dataset/metadata.csv") -lt 100 ]; then
#     echo "Curating multimodal face dataset (>= 10,000 images)..."
#     python3 data/curate_dataset.py --data-dir dataset --target-count 15000
# else
#     echo "Curated dataset already present with $(wc -l < dataset/metadata.csv) records."
# fi

# Step 3: Launch Training
echo "Launching training..."
python3 train.py \
    --metadata-csv dataset/metadata.csv \
    --vocab-path dataset/vocab.json \
    --output-dir checkpoints \
    --epochs 200 \
    --batch-size 64 \
    --lr 2e-4 \
    --image-size 64 \
    --precision auto \
    --ema-decay 0.9999 \
    --sample-every 5

echo "=== Job Completed successfully at $(date) ==="
