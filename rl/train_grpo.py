"""GRPO training for recommendation-oriented RL (RL-2, see RL_DESIGN.md 方案B).

Uses TRL's GRPOTrainer + LoRA + our graded reward (rl/reward.py). This is the
FRAMEWORK / entry point — the report's two stabilizers (stage-wise clipping, negative-
sample down-weighting) need a GRPOTrainer subclass and are marked TODO below; the base
GRPO loop with the graded reward is fully wired.

⚠️ NOT YET RUN. Intended for A100. Before running:
  - resolve TRL import (env currently errors: `No module named 'mergekit'` -> `pip install mergekit`)
  - decide thinking vs non-thinking start checkpoint (RL_DESIGN.md §0.2)

Probe (rl/rollout.py) confirmed: exact-hit 0.8% (too sparse) but GRADED signal 31% ->
GRPO with accuracy="graded" is the viable path.

Run (once TRL fixed, on A100):
  python train_grpo.py \
      --prompts ../datasets/rl_prompts_tuijian.jsonl \
      --init-adapter ../LLaMA-Factory/saves/onereason-0.8b/lora/exp4 \
      --output ../LLaMA-Factory/saves/onereason-0.8b/grpo/run1 \
      --num-generations 16 --lr 1e-6
"""

import argparse
import json

BASE = "/home/lab/wy/LLM_REC/OneReason-0.8B-pretrain-competition"


def load_dataset(path):
    from datasets import Dataset
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    # GRPOTrainer needs a "prompt" column; extra columns (gold) are passed to reward_func.
    return Dataset.from_list([{"prompt": r["prompt"], "gold": r["gold"]} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="/home/lab/wy/LLM_REC/datasets/rl_prompts_tuijian.jsonl")
    ap.add_argument("--init-adapter", default=None,
                    help="warm-start LoRA adapter (e.g. exp4). Omit for fresh LoRA on base.")
    ap.add_argument("--output", default="/home/lab/wy/LLM_REC/LLaMA-Factory/saves/onereason-0.8b/grpo/run1")
    ap.add_argument("--accuracy", choices=["graded", "exact", "recall"], default="graded",
                    help="reward accuracy component; graded is the viable one (probe: 31% signal)")
    ap.add_argument("--num-generations", type=int, default=16, help="G rollouts per prompt (group size)")
    ap.add_argument("--lr", type=float, default=1e-6, help="GRPO LR (low! itemic entropy is fragile)")
    ap.add_argument("--max-completion-length", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=8)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel
    from trl import GRPOTrainer, GRPOConfig

    from reward import make_grpo_reward_func

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    if args.init_adapter:
        # warm-start from an SFT LoRA, then continue training it under GRPO
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        peft_config = None
    else:
        # IRON RULE: lora_target on attn/mlp only -> embed_tokens/lm_head stay FROZEN
        # (preserves itemic distribution entropy; do NOT add them to target_modules).
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    dataset = load_dataset(args.prompts)
    reward_func = make_grpo_reward_func(accuracy=args.accuracy, diversity=True)

    config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.num_generations,  # one prompt's group per step (adjust to VRAM)
        num_generations=args.num_generations,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        report_to="none",
        # NOTE: TRL exposes epsilon (clip) as a single value. The report's STAGE-WISE clipping
        # (loose on CoT tokens, TIGHT on itemic tokens to prevent entropy collapse) is NOT
        # expressible via config alone.
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # ============================ TODO (A100, report §6 稳定器) ============================
    # 1. STAGE-WISE CLIPPING (Fig 12a): subclass GRPOTrainer, override the loss so itemic
    #    tokens (id >= 151669, see reward.ITEM_* / eval_split_loss_recall.ITEMIC_MIN) use a
    #    TIGHTER clip range (e.g. 0.1/0.15) than CoT/text tokens (0.2/0.28). This is THE
    #    guard against itemic entropy collapse (our exp2 failure mode).
    # 2. NEGATIVE-SAMPLE DOWN-WEIGHTING (eq13): weight non-hit rollouts by β<1 so the ~69%
    #    zero-reward rollouts don't dominate the gradient. Apply per-rollout weight to the loss.
    # Both require overriding compute_loss; base GRPO loop below works without them but is
    # less stable under sparse reward.
    # =====================================================================================

    trainer.train()
    trainer.save_model(args.output)
    print(f"[OK] GRPO done -> {args.output}. Then: div_sa guardrail + official eval to confirm.")


if __name__ == "__main__":
    main()
