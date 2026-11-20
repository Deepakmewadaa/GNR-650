#!/usr/bin/env bash
#SBATCH --job-name=evaluate_different_domains
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

python test.py \
    --checkpoint checkpoints/best_model.pt \
    --test-prompts data/different_domains.csv \
    --output-dir outputs/grader_results_different_domains \
    --steps 14 \
    --cfg-scale 4.5