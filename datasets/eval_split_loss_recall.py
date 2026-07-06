"""Sampling-based Pass@K / Recall@K evaluator for OneReason checkpoints.

WHY THIS EXISTS (read PROGRESS.md §8): the competition 懂推荐 (R3 Recommendation)
metric is Pass@K / Recall@K over a SAMPLED SET of candidates (tech report §3.2 /
appendix B.4), where candidate DIVERSITY is essential. The older
eval_split_loss.py (teacher-forced itemic_loss + greedy Pass@1) is ANTI-correlated
with this: lower cross-entropy = peakier distribution = less sampling diversity =
worse Recall@K. exp2 looked "better" on the old proxy but crashed on the real eval
(v1.0.3 = 0.7212) due to itemic-token entropy collapse.

This tool samples K candidates per example (temperature sampling), extracts itemic
tokens, and reports, per category:
  - Pass@k   : gold itemic token appears in ANY of the k sampled candidates
  - Recall@k : fraction of gold itemic tokens covered by the k candidates
  - div_tok  : avg # distinct itemic tokens generated across K samples (diversity)
  - div_sa   : avg # distinct first sub-token (s_a) values (the report's R_div basis)

A GOOD checkpoint for recommendation has HIGH Pass@K AND HIGH diversity. A collapsed
one (e.g. exp2) has low diversity even if greedy Pass@1 looks OK.

Usage:
  python eval_split_loss_recall.py --adapter saves/onereason-0.8b/lora/exp1 --k 32 --limit 90
  python eval_split_loss_recall.py --base-only --k 32
  python eval_split_loss_recall.py --model-path saves/.../exp2_merged_full --k 32
"""

import argparse
import json
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/lab/wy/LLM_REC/OneReason-0.8B-pretrain-competition"
EVAL_SET = "/home/lab/wy/LLM_REC/datasets/eval_set.jsonl"
# full itemic pattern: <|domain_begin|><s_a_N><s_b_N><s_c_N>
ITEM_RE = re.compile(r"<\|\w+?_begin\|><s_a_(\d+)><s_b_\d+><s_c_\d+>")
ITEM_FULL_RE = re.compile(r"<\|\w+?_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


def gold_items(text):
    return set(ITEM_FULL_RE.findall(text))


def build_prefix(tokenizer, system, instruction):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": instruction}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def sample_candidates(model, tokenizer, prefix, k, temperature, top_p, max_new_tokens):
    inputs = tokenizer([prefix], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=k,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gens = []
    plen = len(inputs.input_ids[0])
    for seq in out:
        gens.append(tokenizer.decode(seq[plen:], skip_special_tokens=True))
    return gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--model-path", default=None, help="eval a full merged model dir directly")
    ap.add_argument("--k", type=int, default=32, help="number of sampled candidates per example")
    ap.add_argument("--k-report", type=int, nargs="+", default=[1, 8, 32],
                    help="Pass@k / Recall@k values to report (must be <= --k)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=90, help="total eval examples (balanced across categories)")
    ap.add_argument("--categories", nargs="+", default=["tuijian", "wuliao", "yonghu"])
    args = ap.parse_args()

    load_from = args.model_path or BASE
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = AutoModelForCausalLM.from_pretrained(load_from, dtype=torch.bfloat16, device_map="cuda")
    tag = f"MERGED:{args.model_path}" if args.model_path else "BASE(no adapter)"
    if not args.base_only and not args.model_path and args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        tag = args.adapter
    model.eval()

    rows = [json.loads(l) for l in open(EVAL_SET, encoding="utf-8")]
    # balance across categories up to limit
    per_cat = max(1, args.limit // len(args.categories))
    picked = []
    seen = defaultdict(int)
    for r in rows:
        c = r["category"]
        if c in args.categories and seen[c] < per_cat and gold_items(r["output"]):
            picked.append(r)
            seen[c] += 1

    ks = sorted(args.k_report)
    agg = defaultdict(lambda: {**{f"pass@{k}": 0 for k in ks}, **{f"rec@{k}": 0.0 for k in ks},
                              "n": 0, "div_tok": 0.0, "div_sa": 0.0})

    for i, r in enumerate(picked):
        prefix = build_prefix(tokenizer, r["system"], r["instruction"])
        gold = gold_items(r["output"])
        gens = sample_candidates(model, tokenizer, prefix, args.k, args.temperature, args.top_p, args.max_new_tokens)
        # per-sample extracted itemic tokens (order preserved for Pass@k prefixes)
        gen_tokens = [gold_items(g) for g in gens]
        union_all = set().union(*gen_tokens) if gen_tokens else set()
        sa_all = set().union(*[set(ITEM_RE.findall(g)) for g in gens]) if gens else set()

        for key in (r["category"], "ALL"):
            a = agg[key]
            a["n"] += 1
            a["div_tok"] += len(union_all)
            a["div_sa"] += len(sa_all)
            for k in ks:
                union_k = set().union(*gen_tokens[:k]) if gen_tokens[:k] else set()
                a[f"pass@{k}"] += 1 if (gold & union_k) else 0
                a[f"rec@{k}"] += len(gold & union_k) / len(gold)

        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(picked)}")

    print(f"\n===== sampling eval: {tag}  (K={args.k}, T={args.temperature}, top_p={args.top_p}) =====")
    cols = "".join([f"Pass@{k:<5}" for k in ks]) + "".join([f"Rec@{k:<6}" for k in ks]) + f"{'div_tok':>9}{'div_sa':>8}"
    print(f"{'category':<10}{cols}  (n)")
    for key in list(args.categories) + ["ALL"]:
        a = agg[key]
        if a["n"] == 0:
            continue
        n = a["n"]
        parts = "".join([f"{a[f'pass@{k}']/n*100:6.1f}% " for k in ks])
        parts += "".join([f"{a[f'rec@{k}']/n*100:6.1f}% " for k in ks])
        parts += f"{a['div_tok']/n:8.2f}{a['div_sa']/n:8.2f}"
        print(f"{key:<10}{parts}  ({n})")


if __name__ == "__main__":
    main()
