#!/usr/bin/env bash
# LoRA SFT of OneReason-0.8B on the onerec_sft dataset (86,326 examples: original 32,480
# + item-grounding/rec/general augmentation from the raw Explorer_LLM_Rec_Competition dataset).
# 1 epoch, saves adapter every 500 steps, auto-restores the best (lowest eval_loss) checkpoint.
# After this finishes, run merge_lora.sh (point it at saves/onereason-0.8b/lora/sft_v3) to
# produce the full merged model.safetensors.
set -euo pipefail

source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate onerec

cd /home/lab/wy/LLM_REC/LLaMA-Factory

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

llamafactory-cli train examples/train_lora/onereason_lora_sft.yaml
