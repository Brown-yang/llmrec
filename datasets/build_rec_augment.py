import glob
import json
import random

import pandas as pd

random.seed(2026)

UP_DIR = "/home/lab/wy/LLM_REC/datasets/OpenOneRec/Explorer_LLM_Rec_Competition/data/OneReason_UserProfile"
SID_DIR = "/home/lab/wy/LLM_REC/datasets/OpenOneRec/Explorer_LLM_Rec_Competition/data/OneReason_Pid2Sid"
OUT_PATH = "/home/lab/wy/LLM_REC/datasets/dataset_orin/懂推荐_augmented.jsonl"

USERS_PER_FILE = 2500  # 10 files -> ~25,000 sampled users
MAX_HISTORY_PER_DOMAIN = 50

# ---- domain field plan: (history_field, target_field(s), user_domain_key, token_prefix) ----
DOMAIN_PLAN = {
    "video": {
        "history_field": "video_history_sampled_pid_list",
        "recent_field": "video_sampled_pid_list",  # extra context + fallback target pool
        "up_domain": "video/video",
        "token": "video",
        "resp_template": "该用户最近喜欢的视频有: {tok}",
        "weight": 0.60,
    },
    "goods": {
        # BUG FIX: was ec_colossus_rs_item_id_list (system-shown candidates, 86.8% never
        # clicked -> noise labelled as "浏览"). Now use REAL clicks as the history, and the
        # last click as the target (matches official response "点击了商品"). This drops the
        # colossus_rs noise entirely. See PROGRESS.md §6.
        "history_field": "ec_good_click_item_id_list_extend",  # real clicks
        "recent_field": None,
        "target_field": None,  # -> target = last click (held out from history via fallback)
        "up_domain": "goods",
        "token": "prod",
        "resp_template": "该用户最近点击了商品: {tok}",
        "weight": 0.15,
    },
    "live": {
        "history_field": "live_hist_author_id_list",
        "up_domain": "live",
        "token": "living",
        "resp_template": "该用户最近首次打赏了主播: {tok}",
        "weight": 0.12,
    },
    "ad": {
        "history_field": "outer_loop_history_action_pid_list_click",
        "target_field": "outer_loop_deep_target_pid",
        "up_domain": "video/ad",
        "token": "ad",
        "resp_template": "该用户最近感兴趣的广告有: {tok}",
        "weight": 0.13,
    },
}

SYSTEMS = [
    "你擅长理解快手用户跨场景行为和语义ID表示，请根据输入信息归纳该用户的目标内容。",
    "你是推荐理解助手。你需要根据用户多域历史行为，输出该用户在各推荐场景中的目标内容",
    "你负责根据用户多域行为理解用户兴趣偏好，并输出该用户在各场景中的目标内容。",
    "你要把用户历史行为转换成推荐目标描述，请保持输出简洁、准确，并覆盖所有非空场景。",
    "你是一名多场景推荐数据构造助手，请结合用户行为线索，生成对应的目标内容文本。",
    "你是一位推荐系统分析助手，请阅读用户历史行为，并给出直播、电商、视频、广告场景的",
]

TAIL_INSTRUCTIONS = [
    "请根据以上信息，给出该用户在直播、电商、视频、广告场景中的目标内容。",
    "请输出该用户在不同场景下对应的目标内容。",
    "请基于这些线索总结该用户在各场景中的目标内容。",
]

DOMAIN_TEXT_TEMPLATES = {
    "video": "用户视频行为: 深度观看了 {hist}{recent_part}",
    "goods": "用户购物行为: 点击了商品 {hist}{recent_part}",
    "live": "用户在直播域: 关注了主播 {hist}",
    "ad": "用户广告行为: 点击了广告 {hist}",
}


def sid_tok(token, sid):
    a, b, c = int(sid[0]), int(sid[1]), int(sid[2])
    return f"<|{token}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def get_list(row, field):
    if field is None:
        return []
    v = row.get(field)
    if v is None or not hasattr(v, "__len__"):
        return []
    return list(v)


# ---------- Pass 1: sample users, collect needed (domain, pid) pairs ----------
up_files = sorted(glob.glob(f"{UP_DIR}/*.parquet"))
sampled_rows = []  # list of dict(row_fields...)
needed_pids = {}  # up_domain -> set(pid)
for up_dom in {p["up_domain"] for p in DOMAIN_PLAN.values()}:
    needed_pids[up_dom] = set()

all_fields = set()
for plan in DOMAIN_PLAN.values():
    for k in ("history_field", "recent_field", "target_field"):
        if plan.get(k):
            all_fields.add(plan[k])

for fp in up_files:
    df = pd.read_parquet(fp, columns=list(all_fields))
    df = df.sample(n=min(USERS_PER_FILE, len(df)), random_state=2026)
    for _, row in df.iterrows():
        rec = {}
        for dom, plan in DOMAIN_PLAN.items():
            hist = get_list(row, plan.get("history_field"))[-MAX_HISTORY_PER_DOMAIN:]
            recent = get_list(row, plan.get("recent_field"))[-MAX_HISTORY_PER_DOMAIN:] if plan.get("recent_field") else []
            target_pool = get_list(row, plan.get("target_field")) if plan.get("target_field") else []
            rec[dom] = {"hist": hist, "recent": recent, "target_pool": target_pool}
            up_dom = plan["up_domain"]
            for pid in hist + recent + target_pool:
                needed_pids[up_dom].add(pid)
        sampled_rows.append(rec)

print(f"[INFO] sampled {len(sampled_rows)} users from {len(up_files)} files")
for d, s in needed_pids.items():
    print(f"[INFO] needed pids for domain {d}: {len(s)}")

# ---------- Pass 2: resolve pid -> sid via Pid2Sid ----------
sid_files = sorted(glob.glob(f"{SID_DIR}/*.parquet"))
pid2sid = {up_dom: {} for up_dom in needed_pids}
for i, fp in enumerate(sid_files):
    df = pd.read_parquet(fp, columns=["pid", "domain", "sid_three"])
    for pid, dom, sid in zip(df["pid"], df["domain"], df["sid_three"]):
        if dom in needed_pids and pid in needed_pids[dom] and pid not in pid2sid[dom]:
            pid2sid[dom][pid] = sid
    if (i + 1) % 40 == 0:
        print(f"[INFO] Pid2Sid scan {i+1}/{len(sid_files)}")

for d, m in pid2sid.items():
    print(f"[INFO] resolved sid for domain {d}: {len(m)} / {len(needed_pids[d])}")

# ---------- Pass 3: build alpaca-style records ----------
records = []
for rec in sampled_rows:
    domain_blocks = []
    candidate_targets = []  # (domain_key, token_str, weight)

    for dom, plan in DOMAIN_PLAN.items():
        up_dom = plan["up_domain"]
        sidmap = pid2sid[up_dom]
        hist_tokens = [sid_tok(plan["token"], sidmap[p]) for p in rec[dom]["hist"] if p in sidmap]
        recent_pids = rec[dom]["recent"]
        target_pool = rec[dom]["target_pool"]

        if not hist_tokens and not recent_pids:
            continue

        # decide target: prefer explicit target_field, else last of "recent", else last of hist
        target_pid = None
        for p in reversed(target_pool):
            if p in sidmap:
                target_pid = p
                break
        if target_pid is None:
            for p in reversed(recent_pids):
                if p in sidmap and p not in rec[dom]["hist"][-1:]:
                    target_pid = p
                    break
        if target_pid is None and hist_tokens:
            # fall back to holding out the very last history item as target
            for p in reversed(rec[dom]["hist"]):
                if p in sidmap:
                    target_pid = p
                    hist_tokens = [sid_tok(plan["token"], sidmap[q]) for q in rec[dom]["hist"] if q in sidmap and q != p]
                    break

        recent_tokens = [sid_tok(plan["token"], sidmap[p]) for p in recent_pids if p in sidmap and p != target_pid]

        if hist_tokens or recent_tokens:
            recent_part = f"，看过 {', '.join(recent_tokens)}" if recent_tokens else ""
            text = DOMAIN_TEXT_TEMPLATES[dom].format(hist=", ".join(hist_tokens) if hist_tokens else "(无)", recent_part=recent_part)
            domain_blocks.append(text)

        if target_pid is not None:
            tok = sid_tok(plan["token"], sidmap[target_pid])
            candidate_targets.append((dom, tok, plan["weight"]))

    if not domain_blocks or not candidate_targets:
        continue

    # pick target domain weighted similar to the original corpus's domain distribution
    doms, toks, weights = zip(*candidate_targets)
    target_idx = random.choices(range(len(doms)), weights=weights, k=1)[0]
    target_dom, target_tok = doms[target_idx], toks[target_idx]

    prompt = (
        "以下是一个用户的多域历史行为信息：\n"
        + "\n".join(domain_blocks)
        + "\n\n"
        + random.choice(TAIL_INSTRUCTIONS)
        + " /no_think"
    )
    system = random.choice(SYSTEMS)
    response = DOMAIN_PLAN[target_dom]["resp_template"].format(tok=target_tok)

    records.append([{"system": system, "prompt": prompt, "response": response}])

random.shuffle(records)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[OK] wrote {len(records)} augmented recommendation examples to {OUT_PATH}")
from collections import Counter
print(Counter(r[0]["response"].split(":")[0].split("：")[0] for r in records))
