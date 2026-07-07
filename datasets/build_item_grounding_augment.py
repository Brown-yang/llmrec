"""FIXED item-grounding augmentation (懂物料, caption->token direction).

Fixes the misalignments that made the original backfire (exp4: 懂物料 0.1533->0.1226):
  1. CAPTION CONTENT: original used Pid2Caption verbatim -> 46% were short title/tag-lists
     (min 12 chars) while official captions are rich 187-294-char prose. Now FILTER to
     rich descriptions only (len in [MIN_CAP, MAX_CAP], not a stringified tag-list).
  2. TEMPLATE ALIGN: instead of hand-written system/prompt variants, SAMPLE the real
     (system, prompt-prefix, marker) tuples straight from the official 懂物料 caption->token
     samples, per domain. Guarantees exact format alignment with training/eval.
  3. JOIN: join Pid2Caption <-> Pid2Sid on (domain, pid) (README requires it), not pid alone.
     sid_three is a float array [a., b., c.] -> cast to int tokens <s_a_a><s_b_b><s_c_c>.
  4. REPRODUCIBLE: reads parquet directly (no lost /tmp pickle).

Domains: Pid2Caption/Sid have {goods, video/video, video/ad} -> tokens {prod, video, ad}.
(live has no captions in Pid2Caption, so live grounding is not augmented here.)
"""

import glob
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd

random.seed(2026)

DATA = "/home/lab/wy/LLM_REC/datasets/OpenOneRec/Explorer_LLM_Rec_Competition/data"
OFFICIAL_GLOB = "/home/lab/wy/LLM_REC/datasets/dataset_orin/懂物料part*.jsonl"
OUT = "/home/lab/wy/LLM_REC/datasets/dataset_orin/懂物料_augmented.jsonl"

DOMAIN2TOKEN = {"goods": "prod", "video/video": "video", "video/ad": "ad"}
TOKEN2DOMAIN = {v: k for k, v in DOMAIN2TOKEN.items()}
MIN_CAP, MAX_CAP = 150, 400   # official captions are 187-294 chars; keep rich prose only
TARGET = 30000                # cap final records

ITEM_RE = re.compile(r"<\|(\w+?)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


def is_rich(cap) -> bool:
    """Rich natural-language description like official (not a short title / tag-list)."""
    if not isinstance(cap, str):
        return False
    c = cap.strip()
    if c.startswith("[") and c.endswith("]"):
        return False  # stringified tag-list
    return MIN_CAP <= len(c) <= MAX_CAP


# ---------- 1. extract REAL official caption->token templates, per token-domain ----------
templates = defaultdict(list)  # tokdom(prod/video/ad) -> list of (system, prefix, marker)
for f in glob.glob(OFFICIAL_GLOB):
    for line in open(f, encoding="utf-8"):
        o = json.loads(line)[0]
        resp = o.get("response", "") or ""
        m = ITEM_RE.search(resp)
        if not m:
            continue  # skip token->caption direction; we only augment caption->token
        tokdom = m.group(1)
        if tokdom not in TOKEN2DOMAIN:
            continue
        p = o.get("prompt", "") or ""
        if "：" not in p:
            continue
        prefix = p.split("：", 1)[0]
        marker = ""
        for mk in ("/no_think", "/think"):
            if p.rstrip().endswith(mk):
                marker = mk
                break
        templates[tokdom].append((o.get("system", ""), prefix, marker))

for d, t in templates.items():
    print(f"[templates] {d}: {len(t)} official (system,prefix,marker) samples")
assert templates, "no official caption->token templates found"


# ---------- 2. read Pid2Caption -> rich (domain,pid)->caption, BALANCED per domain ----------
# caption files are domain-ordered, so cap each domain independently to avoid a video-only set
CAP_PER_DOM = TARGET // len(DOMAIN2TOKEN) + 3000   # ~13k per domain -> ~30k balanced output
per_dom = defaultdict(dict)
for fp in sorted(glob.glob(f"{DATA}/OneReason_Pid2Caption/*.parquet")):
    df = pd.read_parquet(fp, columns=["pid", "domain", "caption"])
    for pid, dom, cap in zip(df["pid"], df["domain"], df["caption"]):
        if dom in DOMAIN2TOKEN and len(per_dom[dom]) < CAP_PER_DOM and is_rich(cap):
            per_dom[dom][(dom, pid)] = cap.strip()
    if all(len(per_dom[d]) >= CAP_PER_DOM for d in DOMAIN2TOKEN):
        break
needed = {k: v for d in per_dom for k, v in per_dom[d].items()}
print(f"[caption] collected rich items per domain: " +
      ", ".join(f"{d}={len(per_dom[d])}" for d in DOMAIN2TOKEN))


# ---------- 3. join with Pid2Sid on (domain,pid) ----------
sid = {}
need_keys = set(needed)
for fp in sorted(glob.glob(f"{DATA}/OneReason_Pid2Sid/*.parquet")):
    df = pd.read_parquet(fp, columns=["pid", "domain", "sid_three"])
    for pid, dom, s3 in zip(df["pid"], df["domain"], df["sid_three"]):
        k = (dom, pid)
        if k in need_keys and k not in sid:
            a, b, c = (int(round(float(x))) for x in list(s3)[:3])
            sid[k] = (a, b, c)
    if len(sid) >= len(need_keys):
        break
print(f"[sid] resolved sid for {len(sid)} / {len(need_keys)} items")


# ---------- 4. build records with REAL official templates ----------
records = []
for k, cap in needed.items():
    if k not in sid:
        continue
    tokdom = DOMAIN2TOKEN[k[0]]
    if tokdom not in templates:
        continue
    system, prefix, marker = random.choice(templates[tokdom])
    a, b, c = sid[k]
    prompt = f"{prefix}：{cap}{marker}"
    response = f"<|{tokdom}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"
    records.append([{"system": system, "prompt": prompt, "response": response}])

random.shuffle(records)   # shuffle BEFORE truncating so the TARGET cut stays domain-balanced
records = records[:TARGET]
with open(OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[OK] wrote {len(records)} ALIGNED item-grounding examples -> {OUT}")
print("  domain分布:", Counter(r[0]["response"].split("<s_a")[0].split("|")[1] for r in records))
lens = sorted(len(r[0]["prompt"]) for r in records)
print(f"  prompt长度中位: {lens[len(lens)//2] if lens else 0}(对齐官方rich描述)")
