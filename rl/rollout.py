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

from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import extract_items, extract_sa, itemic_hit, diversity_factor, graded_reward, parse_item

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
    # per-domain stats keyed by the gold item's domain (video/prod/ad/living)
    dom = defaultdict(lambda: {"n": 0, "hit_prompts": 0, "hits": 0,
                               "graded_signal_prompts": 0, "graded_sum": 0.0, "div_sum": 0.0})

    def gold_domain(gold):
        p = parse_item(gold[0]) if gold else None
        return p[0] if p else "?"

    for i, r in enumerate(rows):
        comps = sample(model, tokenizer, r["prompt"], args.n,
                       args.temperature, args.top_p, args.max_new_tokens)
        gold = r["gold"]
        d = gold_domain(gold)
        hits = [c for c in comps if itemic_hit(c, gold) == 1.0]
        best_graded = max((graded_reward(c, gold) for c in comps), default=0.0)

        s = dom[d]
        s["n"] += 1
        s["hit_prompts"] += 1 if hits else 0
        s["hits"] += len(hits)
        s["graded_signal_prompts"] += 1 if best_graded > 0 else 0  # got any partial credit
        s["graded_sum"] += best_graded
        s["div_sum"] += diversity_factor(comps)

        if hit_dump is not None:
            for c in hits:
                hit_dump.write(json.dumps({
                    "instruction": r["instruction"], "input": "",
                    "output": c.strip(), "system": r.get("system", ""),
                }, ensure_ascii=False) + "\n")

        if (i + 1) % 20 == 0:
            t = sum(v["n"] for v in dom.values())
            hp = sum(v["hit_prompts"] for v in dom.values())
            gp = sum(v["graded_signal_prompts"] for v in dom.values())
            print(f"  ...{i+1}/{len(rows)}  exact-Pass@{args.n}={hp/t*100:.1f}%  graded-signal={gp/t*100:.1f}%")

    if hit_dump is not None:
        hit_dump.close()

    print(f"\n===== rollout stats (N={args.n}, T={args.temperature}) =====")
    print(f"{'domain':<8}{'n':>5}{'exactPass@N':>13}{'gradedSignal':>14}{'avgGraded':>11}{'avgDiv':>9}")
    tot = {k: 0 for k in ("n", "hit_prompts", "graded_signal_prompts", "hits")}
    tot.update({"graded_sum": 0.0, "div_sum": 0.0})
    for d, s in sorted(dom.items()):
        for k in tot:
            tot[k] += s[k]
        print(f"{d:<8}{s['n']:>5}{s['hit_prompts']/s['n']*100:>12.1f}%"
              f"{s['graded_signal_prompts']/s['n']*100:>13.1f}%"
              f"{s['graded_sum']/s['n']:>11.3f}{s['div_sum']/s['n']:>9.3f}")
    n = tot["n"]
    print(f"{'ALL':<8}{n:>5}{tot['hit_prompts']/n*100:>12.1f}%"
          f"{tot['graded_signal_prompts']/n*100:>13.1f}%{tot['graded_sum']/n:>11.3f}{tot['div_sum']/n:>9.3f}")
    if args.dump_hits:
        print(f"\n[RFT] wrote {tot['hits']} exact-hit rollouts -> {args.dump_hits}")


if __name__ == "__main__":
    main()
