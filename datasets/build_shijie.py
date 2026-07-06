"""Build 懂世界 (common_sense) SFT data — the ONE dimension the official competition
data has ZERO of (see PROGRESS.md). Format reverse-engineered from the eval logs
(logs/*.log, task challenge_common_sense):

  system : 你是一个非常聪明的助手，请直接遵循指示作答。
  user   : 请回答以下问题：
           {question}
           A.{a}
           B.{b}
           C.{c}
           D.{d}
           请按以下格式作答："正确答案是 (在此处填写选项字母)"
  answer : 正确答案是 {LETTER}     (default; --bare gives just "{LETTER}")

Eval domain observed = Chinese MCQ mixing 常识(地理/生活) + 数学(代数/应用题) + 逻辑推理(奥数).
Source: CMMLU (67 Chinese subjects, ~11.5k Q) matches best; also supports a generic MCQ jsonl.

Output: dataset_orin/懂世界.jsonl in the same list-wrapped alpaca format as the other
official files (`[{"system":..,"prompt":..,"response":..}]` per line) so prepare_sft_data.py
consumes it automatically.
"""

import argparse
import json
import os
import random

SYSTEM = "你是一个非常聪明的助手，请直接遵循指示作答。"
INSTR_HEAD = "请回答以下问题："
INSTR_TAIL = '请按以下格式作答："正确答案是 (在此处填写选项字母)"'
LETTERS = ["A", "B", "C", "D", "E", "F"]


def make_record(question: str, options: list[str], answer_letter: str, bare: bool):
    """options = list of choice texts; answer_letter in A..; returns list-wrapped alpaca record."""
    opt_lines = "\n".join(f"{LETTERS[i]}.{o}" for i, o in enumerate(options))
    prompt = f"{INSTR_HEAD}\n{question.strip()}\n{opt_lines}\n{INSTR_TAIL}"
    answer_letter = answer_letter.strip().upper()
    response = answer_letter if bare else f"正确答案是 {answer_letter}"
    return [{"system": SYSTEM, "prompt": prompt, "response": response}]


def from_cmmlu(bare: bool, limit: int | None):
    """Download CMMLU raw CSVs (test+dev, 67 Chinese subjects) and yield records.
    CMMLU ships a dataset script (unsupported by new `datasets`), so pull CSVs directly."""
    import glob
    import zipfile
    import pandas as pd
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download(repo_id="haonan-li/cmmlu", repo_type="dataset",
                         filename="cmmlu_v1_0_1.zip")
    extract_dir = os.path.join(os.path.dirname(zp), "cmmlu_extracted")
    if not os.path.isdir(extract_dir):
        with zipfile.ZipFile(zp) as z:
            z.extractall(extract_dir)
    csvs = sorted(glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True))
    print(f"[cmmlu] {len(csvs)} csv files")
    n = 0
    for fp in csvs:
        df = pd.read_csv(fp)
        cols = {c.lower(): c for c in df.columns}
        qc = cols.get("question"); ac = cols.get("answer")
        if not qc or not ac:
            continue
        for _, row in df.iterrows():
            q = str(row[qc]); ans = str(row[ac]).strip().upper()
            opts = [str(row[cols[k]]) for k in ("a", "b", "c", "d") if k in cols]
            if not q or ans not in LETTERS or len(opts) < 4:
                continue
            yield make_record(q, opts, ans, bare)
            n += 1
            if limit and n >= limit:
                return


def _num_distractors(n: float, k: int = 3):
    """Plausible wrong numeric options near the correct answer n (common arithmetic slips)."""
    is_int = float(n).is_integer()
    n = int(n) if is_int else float(n)
    cand = []
    for d in [n * 2, n / 2, n + 1, n - 1, n + 2, n - 2, n + 5, n - 5,
              n + 10, n - 10, int(n * 1.5) if is_int else n * 1.5, n * 3]:
        d = int(d) if is_int else round(float(d), 2)
        if d != n and d > 0 and d not in cand:
            cand.append(d)
    random.shuffle(cand)
    out = []
    for c in cand:
        if len(out) >= k:
            break
        out.append(c)
    return out


def from_gsm8k_zh(path: str, bare: bool, limit: int | None):
    """meta-math/GSM8K_zh: Chinese math word problems w/ numeric answers -> 4-choice MCQ.
    Correct answer + 3 plausible numeric distractors (eval math is MCQ, e.g. 路灯 A.55 B.54...)."""
    random.seed(2026)
    data = json.load(open(path, encoding="utf-8"))
    n = 0
    for row in data:
        q = row.get("question_zh") or row.get("question")
        raw = str(row.get("answer_only", "")).strip().replace(",", "")
        if not q:
            continue
        try:
            correct = float(raw)
        except ValueError:
            continue
        dist = _num_distractors(correct, 3)
        if len(dist) < 3:
            continue
        correct_disp = int(correct) if float(correct).is_integer() else correct
        opts = [correct_disp] + dist
        random.shuffle(opts)
        ans = LETTERS[opts.index(correct_disp)]
        yield make_record(q, [str(o) for o in opts], ans, bare)
        n += 1
        if limit and n >= limit:
            return


def from_jsonl(path: str, bare: bool, limit: int | None):
    """Generic: each line has {question, options:[..] OR A/B/C/D, answer}."""
    n = 0
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        q = o.get("question") or o.get("Question")
        if "options" in o:
            opts = o["options"]
        else:
            opts = [o[k] for k in ("A", "B", "C", "D") if k in o]
        ans = (o.get("answer") or o.get("Answer") or "").strip().upper()
        if not q or ans not in LETTERS or len(opts) < 2:
            continue
        yield make_record(q, opts, ans, bare)
        n += 1
        if limit and n >= limit:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cmmlu", "jsonl", "gsm8k_zh"], default="cmmlu")
    ap.add_argument("--input", help="for --source jsonl/gsm8k_zh: path to the source file")
    ap.add_argument("--out", default="/home/lab/wy/LLM_REC/datasets/dataset_orin/懂世界.jsonl")
    ap.add_argument("--bare", action="store_true", help="response = bare letter (default: '正确答案是 X')")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    if args.source == "cmmlu":
        gen = from_cmmlu(args.bare, args.limit)
    elif args.source == "gsm8k_zh":
        gen = from_gsm8k_zh(args.input, args.bare, args.limit)
    else:
        gen = from_jsonl(args.input, args.bare, args.limit)
    records = list(gen)
    if args.shuffle:
        random.seed(2026)
        random.shuffle(records)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(records)} 懂世界 records -> {args.out}")
    if records:
        r = records[0][0]
        print("--- 示例 ---")
        print("prompt:", r["prompt"][:180])
        print("response:", r["response"])


if __name__ == "__main__":
    main()
