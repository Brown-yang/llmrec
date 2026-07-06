#!/usr/bin/env bash
# Full-parameter SFT of OneReason-0.8B on the onerec_sft dataset.
# Saves full model weights (bf16 safetensors, ~1.6GB) every 2000 steps, 6000 steps total.
set -euo pipefail

source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate onerec

cd /home/lab/wy/LLM_REC/LLaMA-Factory

# Reduces CUDA memory fragmentation; full fine-tuning holds a lot more optimizer
# state than LoRA did, so this is more important here than it was last time.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

llamafactory-cli train examples/train_full/onereason_full_sft.yaml
