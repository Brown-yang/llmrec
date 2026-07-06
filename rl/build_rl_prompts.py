"""Build the RL prompt set: (prompt, gold_items) pairs for RFT / GRPO / DPO.

Source options (RL_DESIGN.md §0.3):
  - official 懂推荐 data (dataset_orin/懂推荐*.jsonl): high-quality, format-faithful. DEFAULT.
  - (TODO) constructed from UserProfile (500k users) for scale + real-distribution coverage.

Output jsonl, one line per prompt:
  {
    "prompt": "<chat-templated prompt string, ending at assistant generation prompt>",
    "gold":   ["<|video_begin|><s_a_..><s_b_..><s_c_..>", ...],   # ground-truth itemic items
    "system": "...", "instruction": "..."                          # raw pieces (for flexibility)
  }

The gold is parsed from the official response (after stripping the <think> CoT). For RL we
present the prompt in NON-thinking or thinking form depending on --mode (see RL_DESIGN.md §0.2:
RL likely wants thinking; default here is non-thinking to match the champion starting point --
switch with --mode think once we have a thinking-capable RL start checkpoint).
"""

import argparse
import glob
import json
import os
import re

from transformers import AutoTokenizer

from reward import extract_items

SRC_DIR = "/home/lab/wy/LLM_REC/datasets/dataset_orin"
MODEL_PATH = "/home/lab/wy/LLM_REC/OneReason-0.8B-pretrain-competition"
THINK_RE = re.compile(r"^<think>.*?</think>\n?", re.DOTALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/lab/wy/LLM_REC/datasets/rl_prompts_tuijian.jsonl")
    ap.add_argument("--glob", default="懂推荐[0-9]*.jsonl", help="which official files to use")
    ap.add_argument("--mode", choices=["think", "nothink"], default="nothink",
                    help="prompt mode marker; nothink matches champion start, think for RL-on-thinking")
    ap.add_argument("--max-prompt-tokens", type=int, default=4096,
                    help="drop prompts longer than this (keep rollout cost bounded)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    files = sorted(glob.glob(os.path.join(SRC_DIR, args.glob)))

    kept, skipped_nogold, skipped_long = 0, 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if args.limit and kept >= args.limit:
                        break
                    obj = json.loads(line)[0]
                    system = obj.get("system", "") or ""
                    prompt = obj["prompt"].rstrip()
                    resp = THINK_RE.sub("", obj["response"], count=1)
                    gold = list(dict.fromkeys(extract_items(resp)))  # unique, order-preserving
                    if not gold:
                        skipped_nogold += 1
                        continue

                    # normalize the mode marker
                    for suf in ("/think", "/no_think"):
                        if prompt.endswith(suf):
                            prompt = prompt[: -len(suf)].rstrip()
                    prompt = prompt + (" /think" if args.mode == "think" else " /no_think")

                    msgs = ([{"role": "system", "content": system}] if system else []) + [
                        {"role": "user", "content": prompt}
                    ]
                    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    if len(tok(chat)["input_ids"]) > args.max_prompt_tokens:
                        skipped_long += 1
                        continue

                    out.write(json.dumps({
                        "prompt": chat,
                        "gold": gold,
                        "system": system,
                        "instruction": prompt,
                    }, ensure_ascii=False) + "\n")
                    kept += 1

    print(f"[OK] wrote {kept} RL prompts to {args.out} "
          f"(skipped no-gold={skipped_nogold}, too-long={skipped_long}, mode={args.mode})")


if __name__ == "__main__":
    main()
