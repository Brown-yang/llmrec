"""Build a FIXED, held-out evaluation set drawn ONLY from official competition
data (dataset_orin, no augmented files). The same file is reused across every
experiment so itemic-vs-text loss numbers are directly comparable.

Per-category sampling keeps the eval set balanced so a category with many rows
(懂推荐) doesn't dominate the averaged metrics.
"""

import glob
import json
import os
import random
import re

from transformers import AutoTokenizer

SRC_DIR = "/home/lab/wy/LLM_REC/datasets/dataset_orin"
OUT_PATH = "/home/lab/wy/LLM_REC/datasets/eval_set.jsonl"
MODEL_PATH = "/home/lab/wy/LLM_REC/OneReason-0.8B-pretrain-competition"
MAX_TOKENS = 4096
PER_CATEGORY = 300  # sampled per logical category (物料 / 用户 / 推荐)

# logical category -> source-file glob (official files only, no *_augmented)
CATEGORIES = {
    "wuliao": "懂物料part*.jsonl",
    "yonghu": "懂用户.jsonl",
    "tuijian": "懂推荐[0-9]*.jsonl",
}

THINK_RE = re.compile(r"^<think>.*?</think>\n?", re.DOTALL)
TRUNCATION_NOTICE = "（注：更早期的行为记录已省略）"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def token_len(system, instruction, output):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
    )
    if system:
        text = system + text
    return len(tokenizer(text + output)["input_ids"])


def smart_truncate_prompt(system, prompt, response, max_tokens):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    lo, hi = 0, len(prompt_ids)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0:
            candidate_text = ""
        else:
            candidate_text = tokenizer.decode(prompt_ids[-mid:])
            if mid < len(prompt_ids):
                candidate_text = TRUNCATION_NOTICE + candidate_text
        if token_len(system, candidate_text, response) <= max_tokens:
            best = candidate_text
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def process(obj):
    system = obj.get("system", "") or ""
    prompt = obj["prompt"]
    response = THINK_RE.sub("", obj["response"], count=1).lstrip("\n")
    if not response.strip():
        return None
    sp = prompt.rstrip()
    if sp.endswith("/think"):
        prompt = sp[: -len("/think")].rstrip() + " /no_think"
    else:
        prompt = sp
    if token_len(system, prompt, response) > MAX_TOKENS:
        prompt = smart_truncate_prompt(system, prompt, response, MAX_TOKENS)
        if prompt is None:
            return None
    return {"instruction": prompt, "input": "", "output": response, "system": system}


rng = random.Random(12345)
records = []
for cat, pattern in CATEGORIES.items():
    lines = []
    for fp in sorted(glob.glob(os.path.join(SRC_DIR, pattern))):
        with open(fp, encoding="utf-8") as f:
            lines.extend(f.readlines())
    rng.shuffle(lines)
    kept = 0
    for line in lines:
        if kept >= PER_CATEGORY:
            break
        obj = json.loads(line)[0]
        rec = process(obj)
        if rec is None:
            continue
        rec["category"] = cat
        records.append(rec)
        kept += 1
    print(f"[INFO] {cat}: kept {kept} eval examples")

rng.shuffle(records)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[OK] wrote {len(records)} eval examples to {OUT_PATH}")
