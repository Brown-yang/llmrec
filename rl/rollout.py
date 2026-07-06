"""Rollout sampler for RL (shared by RFT / GRPO warm-start).

Given a policy (base + optional LoRA adapter) and an RL prompt set (build_rl_prompts.py),
sample candidate completions and score them with the reward functions (reward.py).

Two modes:
  --stage one   : G independent samples per prompt (simple; matches TRL GRPO's default).
  --stage two   : report's two-stage (N reasoning traces x K itemic each). Approximated by
                  sampling N*K completions but grouping K-by-K for the diversity factor.
                  (True prefix-sharing is a later efficiency optimization.)

Primary use RIGHT NOW = **RFT data mining**: with --dump-hits, writes every rollout whose
itemic tokens hit the gold set to an alpaca-style jsonl, ready to re-SFT with LLaMA-Factory
(RL_DESIGN.md scheme A). Also prints hit-rate / diversity stats so we can gauge whether the
reward signal is strong enough before committing to full GRPO.

⚠️ needs GPU. Not run yet (RL-0 is code-prep). Reward logic (reward.py) is already CPU-tested.
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import extract_items, extract_sa, itemic_hit, diversity_factor

BASE = "/home/lab/wy/LLM_REC/OneReason-0.8B-pretrain-competition"


@torch.no_grad()
def sample(model, tokenizer, prompt, n, temperature, top_p, max_new_tokens):
    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, do_sample=True, temperature=temperature, top_p=top_p,
        num_return_sequences=n, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    plen = len(inputs.input_ids[0])
    return [tokenizer.decode(s[plen:], skip_special_tokens=True) for s in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit = base policy)")
    ap.add_argument("--prompts", default="/home/lab/wy/LLM_REC/datasets/rl_prompts_tuijian.jsonl")
    ap.add_argument("--n", type=int, default=32, help="samples per prompt (G, or N*K)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=200, help="#prompts to roll out")
    ap.add_argument("--dump-hits", default=None, help="write hit rollouts (alpaca jsonl) for RFT")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8")][: args.limit]

    hit_dump = open(args.dump_hits, "w", encoding="utf-8") if args.dump_hits else None
    n_prompts, n_hit_prompts, total_hits, div_sum = 0, 0, 0, 0.0

    for i, r in enumerate(rows):
        comps = sample(model, tokenizer, r["prompt"], args.n,
                       args.temperature, args.top_p, args.max_new_tokens)
        gold = r["gold"]
        hits = [c for c in comps if itemic_hit(c, gold) == 1.0]
        n_prompts += 1
        n_hit_prompts += 1 if hits else 0
        total_hits += len(hits)
        div_sum += diversity_factor(comps)

        if hit_dump is not None:
            for c in hits:
                # reconstruct an alpaca SFT record from the successful rollout
                hit_dump.write(json.dumps({
                    "instruction": r["instruction"],
                    "input": "",
                    "output": c.strip(),
                    "system": r.get("system", ""),
                }, ensure_ascii=False) + "\n")

        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(rows)}  pass@{args.n} so far={n_hit_prompts/n_prompts*100:.1f}%")

    if hit_dump:
        hit_dump.close()
    print(f"\n[rollout] prompts={n_prompts}  Pass@{args.n}={n_hit_prompts/n_prompts*100:.1f}%  "
          f"avg_hits/prompt={total_hits/n_prompts:.2f}  avg_div={div_sum/n_prompts:.3f}")
    if args.dump_hits:
        print(f"[RFT] wrote {total_hits} hit rollouts -> {args.dump_hits} "
              f"(register + SFT with LLaMA-Factory to do rejection-sampling fine-tuning)")


if __name__ == "__main__":
    main()
