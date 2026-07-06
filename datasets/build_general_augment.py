import glob
import json
import random
import re

import pandas as pd

random.seed(2026)

GEN_DIR = "/home/lab/wy/LLM_REC/datasets/OpenOneRec/Explorer_LLM_Rec_Competition/data/OneReason_General"
OUT_PATH = "/home/lab/wy/LLM_REC/datasets/dataset_orin/懂世界_augmented.jsonl"
TARGET_N = 2500

THINK_RE = re.compile(r"<think>.*?</think>\n?", re.DOTALL)


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


files = sorted(glob.glob(f"{GEN_DIR}/*.parquet"))
random.shuffle(files)

records = []
for fp in files:
    if len(records) >= TARGET_N:
        break
    df = pd.read_parquet(fp, columns=["messages"])
    for raw in df["messages"]:
        if len(records) >= TARGET_N:
            break
        try:
            msgs = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue

        system = ""
        user_msg, assistant_msg = None, None
        for m in msgs:
            role = m.get("role")
            text = extract_text(m.get("content"))
            if role == "system" and not system:
                system = text
            elif role in ("user", "human") and user_msg is None:
                user_msg = text
            elif role == "assistant":
                assistant_msg = text  # keep the last one

        if not user_msg or not assistant_msg:
            continue

        assistant_msg = THINK_RE.sub("", assistant_msg).lstrip("\n")
        if not assistant_msg.strip():
            continue

        prompt = user_msg.rstrip() + " /no_think"
        records.append([{"system": system, "prompt": prompt, "response": assistant_msg}])

random.shuffle(records)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[OK] wrote {len(records)} general-domain examples to {OUT_PATH}")
