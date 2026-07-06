"""Split-loss / Pass@1 evaluator for OneReason LoRA checkpoints.

Reports, on the FIXED eval_set.jsonl (official data only), broken down by
category (物料/用户/推荐):
  - text-token loss    : cross-entropy over non-itemic (natural-language) tokens
  - itemic-token loss  : cross-entropy over itemic tokens (<s_a_*>, <s_b_*>,
                         <s_c_*> and <|*_begin|>/<|*_end|> domain markers)
  - Pass@1             : greedy-decode the answer, exact string match

Overall loss (1.3-ish) hides the itemic component behind the much-lower text
component, so this tool is how we tell whether a change (embed/lm_head training,
rank, data) actually moved the part that matters.

Usage:
  python eval_split_loss.py --adapter saves/onereason-0.8b/lora/exp1 [--no-gen] [--limit N]
  python eval_split_loss.py --base-only      # evaluate the raw pretrain model
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/lab/wy/LLM_REC/OneReason-0.8B-pretrain-competition"
EVAL_SET = "/home/lab/wy/LLM_REC/datasets/eval_set.jsonl"
ITEMIC_MIN = 151669  # token ids >= this are itemic sub-tokens or domain begin/end markers
ITEM_RE = re.compile(r"<\|\w+_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


def build_inputs(tokenizer, system, instruction, output):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": instruction}
    ]
    prefix = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix + output + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return prefix, prefix_ids, full_ids


@torch.no_grad()
def per_example_loss(model, prefix_ids, full_ids):
    input_ids = torch.tensor([full_ids], device=model.device)
    labels = torch.tensor([full_ids], device=model.device).clone()
    labels[0, : len(prefix_ids)] = -100
    out = model(input_ids=input_ids)
    logits = out.logits[0, :-1].float()
    tgt = labels[0, 1:]
    mask = tgt != -100
    logits, tgt = logits[mask], tgt[mask]
    losses = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
    is_itemic = tgt >= ITEMIC_MIN
    text_loss = losses[~is_itemic]
    item_loss = losses[is_itemic]
    return (
        losses.sum().item(),
        len(losses),
        item_loss.sum().item(),
        int(is_itemic.sum()),
        text_loss.sum().item(),
        int((~is_itemic).sum()),
    )


@torch.no_grad()
def greedy_answer(model, tokenizer, prefix, max_new_tokens=64):
    inputs = tokenizer([prefix], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for --base-only)")
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--model-path", default=None, help="eval a full merged model dir directly (no adapter)")
    ap.add_argument("--no-gen", action="store_true", help="skip Pass@1 greedy decoding (faster)")
    ap.add_argument("--limit", type=int, default=None)
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
    if args.limit:
        rows = rows[: args.limit]

    from collections import defaultdict

    agg = defaultdict(lambda: {"l": 0.0, "n": 0, "il": 0.0, "inn": 0, "tl": 0.0, "tn": 0, "hit": 0, "gen": 0})

    for i, r in enumerate(rows):
        prefix, prefix_ids, full_ids = build_inputs(tokenizer, r["system"], r["instruction"], r["output"])
        if len(full_ids) > 4096:
            continue
        l, n, il, inn, tl, tn = per_example_loss(model, prefix_ids, full_ids)
        for key in (r["category"], "ALL"):
            a = agg[key]
            a["l"] += l; a["n"] += n; a["il"] += il; a["inn"] += inn; a["tl"] += tl; a["tn"] += tn
        if not args.no_gen:
            gen = greedy_answer(model, tokenizer, prefix)
            hit = int(gen == r["output"].strip())
            for key in (r["category"], "ALL"):
                agg[key]["hit"] += hit; agg[key]["gen"] += 1
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(rows)}")

    print(f"\n===== eval: {tag} =====")
    hdr = f"{'category':<10} {'overall_loss':>12} {'text_loss':>10} {'itemic_loss':>12} {'Pass@1':>8}"
    print(hdr)
    for key in ("wuliao", "yonghu", "tuijian", "ALL"):
        a = agg[key]
        if a["n"] == 0:
            continue
        overall = a["l"] / a["n"]
        text = a["tl"] / a["tn"] if a["tn"] else float("nan")
        item = a["il"] / a["inn"] if a["inn"] else float("nan")
        p1 = f"{a['hit']}/{a['gen']}={a['hit']/a['gen']*100:.1f}%" if a["gen"] else "n/a"
        print(f"{key:<10} {overall:>12.4f} {text:>10.4f} {item:>12.4f} {p1:>8}")


if __name__ == "__main__":
    main()
