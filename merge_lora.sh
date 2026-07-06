#!/usr/bin/env bash
# Merges the trained LoRA adapter (saves/onereason-0.8b/lora/sft_v2) into the base
# OneReason-0.8B weights, producing a full standalone model at
# saves/onereason-0.8b/lora/sft_v2_merged/ (model.safetensors, ~1.6GB).
# Run this only after train_lora_sft.sh has finished.
set -euo pipefail

source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate onerec

cd /home/lab/wy/LLM_REC/LLaMA-Factory

llamafactory-cli export examples/merge_lora/onereason_merge.yaml
