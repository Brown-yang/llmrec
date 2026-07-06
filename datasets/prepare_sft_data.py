import json
import random
import re
import glob
import os
from transformers import AutoTokenizer

SRC_DIR = "/home/lab/wy/LLM_REC/datasets/dataset_orin"
EXCLUDE_AUGMENTED = os.environ.get("EXCLUDE_AUGMENTED") == "1"
# KEEP_COT=1 -> preserve the original <think> traces AND the original /think|/no_think
# markers (natural CoT+unCoT mixture per report Table 17). Default (0) = champion
# behavior: strip CoT, force /no_think.
KEEP_COT = os.environ.get("KEEP_COT") == "1"
# NO_TRUNCATE=1 -> keep ALL samples verbatim, no drop, no smart-truncate. Over-length
# samples are left for LLaMA-Factory's own default handling (minimal-processing /
# raw-replication mode).
NO_TRUNCATE = os.environ.get("NO_TRUNCATE") == "1"
# DROP_LONG=1 -> DROP (skip) samples over MAX_TOKENS entirely, instead of smart-truncating
# them. This replicates the CHAMPION's original behavior. exp1 showed smart-truncate hurt
# 懂推荐 (0.4726->0.4373); champion's drop was better. (Ignored if NO_TRUNCATE=1.)
DROP_LONG = os.environ.get("DROP_LONG") == "1"
OUT_TRAIN = os.environ.get("OUT_TRAIN", "/home/lab/wy/LLM_REC/LLaMA-Factory/data/onerec_sft.jsonl")
MODEL_PATH = "/home/lab/wy/LLM_REC/OneReason-0.8B-pretrain-competition"
MAX_TOKENS = 4096  # matches cutoff_len in the training config; drop the long tail
# instead of letting LLaMA-Factory truncate (it truncates from the front of the
# prompt, which would cut off the trailing instruction in our long-context examples).

THINK_RE = re.compile(r"^<think>.*?</think>\n?", re.DOTALL)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def token_len(system, instruction, output):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
    )
    if system:
        text = system + text
    return len(tokenizer(text + output)["input_ids"])


TRUNCATION_NOTICE = "（注：更早期的行为记录已省略）"


def smart_truncate_prompt(system, prompt, response, max_tokens):
    """Keep the tail of the prompt (where the actual instruction lives) and drop
    the oldest behavior-history entries from the front, instead of discarding the
    whole example. Binary-searches for the longest suffix (in prompt tokens) that
    still fits the budget once re-templated with the system/response."""
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


files = sorted(glob.glob(os.path.join(SRC_DIR, "*.jsonl")))
files = [f for f in files if os.path.basename(f) != "onerec_sft.jsonl"]
if EXCLUDE_AUGMENTED:
    files = [f for f in files if "_augmented" not in os.path.basename(f)]
# EXCLUDE_FILES=substr1,substr2 -> drop any source file whose name contains one of these
# (used for surgical augmentation: e.g. include 懂物料_augmented but drop 懂推荐_augmented).
_excl = [s for s in os.environ.get("EXCLUDE_FILES", "").split(",") if s]
if _excl:
    files = [f for f in files if not any(s in os.path.basename(f) for s in _excl)]
print(f"[INFO] EXCLUDE_AUGMENTED={EXCLUDE_AUGMENTED}, using {len(files)} source files: {[os.path.basename(f) for f in files]}")

records = []
skipped_empty = 0
skipped_long = 0
truncated_count = 0
for fp in files:
    cat = os.path.basename(fp)
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)[0]
            system = obj.get("system", "") or ""
            prompt = obj["prompt"]
            response = obj["response"]

            if KEEP_COT:
                # Keep the original <think> trace and the original /think|/no_think
                # marker as-is (natural CoT+unCoT mixture).
                response = response.lstrip("\n")
                prompt = prompt.rstrip()
            else:
                # champion behavior: strip CoT -> keep only the final answer
                response = THINK_RE.sub("", response, count=1).lstrip("\n")
                # normalize the trailing mode marker to /no_think
                stripped_prompt = prompt.rstrip()
                if stripped_prompt.endswith("/think"):
                    prompt = stripped_prompt[: -len("/think")].rstrip() + " /no_think"
                else:
                    prompt = stripped_prompt

            if not response.strip():
                skipped_empty += 1
                continue

            if not NO_TRUNCATE and token_len(system, prompt, response) > MAX_TOKENS:
                if DROP_LONG:
                    # champion behavior: drop over-length samples entirely
                    skipped_long += 1
                    continue
                truncated = smart_truncate_prompt(system, prompt, response, MAX_TOKENS)
                if truncated is None:
                    skipped_long += 1
                    continue
                prompt = truncated
                truncated_count += 1

            records.append(
                {
                    "instruction": prompt,
                    "input": "",
                    "output": response,
                    "system": system,
                    "_category": cat,
                }
            )

print(
    f"total records: {len(records)}, skipped empty-after-strip: {skipped_empty}, "
    f"smart-truncated (front of prompt cut): {truncated_count}, skipped too-long even after truncation: {skipped_long}"
)

random.seed(42)
random.shuffle(records)

for r in records:
    del r["_category"]

os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"wrote {len(records)} records to {OUT_TRAIN}")
