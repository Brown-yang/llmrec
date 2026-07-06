import ast
import json
import pickle
import random

random.seed(2026)

with open("/tmp/found_captions.pkl", "rb") as f:
    captions = pickle.load(f)  # (domain, a, b, c) -> caption (str, sometimes a stringified list)

DOMAIN2TOKEN = {"video/video": "video", "video/ad": "ad", "goods": "prod", "live": "living"}

SYSTEMS = {
    "goods": [
        "你是一位商品理解专家，可以根据商品描述生成准确的商品token。",
        "作为商品标识生成助手，你需要根据给定的商品描述输出匹配的商品token。",
        "你是商品token生成专家，能够将商品描述转化为对应的结构化商品token。",
        "你具备从商品特征描述中提取关键信息并输出商品token的能力。",
        "作为AI商品标识助手，你可以根据商品描述生成匹配的商品token。",
        "你擅长把商品内容描述映射成精确的商品token。",
    ],
    "live": [
        "你擅长把主播的外在形象、直播内容和风格描述映射成精确的主播token。",
        "作为主播标识生成助手，你需要根据给定的主播描述输出匹配的主播token。",
        "你是一位直播理解专家，可以根据主播画像与内容风格描述生成准确的主播token。",
        "你具备从主播特征描述中提取关键信息并输出主播token的能力。",
        "你是一名专业的主播token生成助手，请根据主播的形象、内容与风格描述生成匹配的主播token。",
        "你是主播token生成专家，能够将主播描述转化为对应的结构化主播token。",
    ],
    "video/ad": [
        "请根据输入的广告描述，输出能与其语义最匹配的广告token。",
        "你是广告语义到token的映射助手，请阅读广告描述并生成准确的广告token。",
        "你擅长根据广告内容、风格和主题描述，输出对应的广告token。",
        "你是一名广告token生成助手，需要根据广告内容描述生成最匹配的广告token。",
    ],
    "video/video": [
        "你是一个短视频语义标识分析助手，能够根据短视频描述生成对应的短视频token。",
        "你是短视频token生成专家，能够将短视频描述转化为对应的结构化短视频token。",
        "你是一名专业的短视频token生成助手，请根据短视频的画面、主体、动作、场景与风格描述生成匹配的短视频token。",
        "作为 AI 短视频标识助手，你可以根据短视频描述生成匹配的短视频token。",
        "你擅长把短视频的内容描述映射成精确的短视频token。",
        "你具备从短视频特征描述中提取关键信息并输出短视频token的能力。",
    ],
}

PROMPT_TEMPLATES = {
    "goods": [
        "下面是一段商品描述，请返回匹配的商品token：{desc}",
        "请从以下商品描述中推断并生成对应的商品token：{desc}",
        "根据以下商品关键信息，生成匹配的商品token：{desc}",
    ],
    "live": [
        "请从以下主播描述中推断并生成对应的主播token：{desc}",
        "请分析这段主播特征描述，并生成对应的主播token：{desc}",
        "根据以下主播特征关键词，生成匹配的主播token：{desc}",
    ],
    "video/ad": [
        "请根据这段广告dense caption生成匹配的广告token：{desc}",
        "请根据以下广告内容描述，生成匹配的广告token：{desc}",
    ],
    "video/video": [
        "基于这段短视频描述，生成最匹配的短视频token：{desc}",
        "请分析这段短视频内容，并生成对应的短视频token：{desc}",
    ],
}


def normalize_caption(cap):
    if isinstance(cap, str) and cap.strip().startswith("[") and cap.strip().endswith("]"):
        try:
            items = ast.literal_eval(cap)
            if isinstance(items, list):
                return "、".join(str(x) for x in items)
        except (ValueError, SyntaxError):
            pass
    return str(cap)


records = []
for (domain, a, b, c), cap in captions.items():
    desc = normalize_caption(cap)
    if not desc.strip():
        continue
    tok = DOMAIN2TOKEN[domain]
    output = f"<|{tok}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"
    system = random.choice(SYSTEMS[domain])
    prompt = random.choice(PROMPT_TEMPLATES[domain]).format(desc=desc) + " /no_think"
    records.append([{"system": system, "prompt": prompt, "response": output}])

random.shuffle(records)

out_path = "/home/lab/wy/LLM_REC/datasets/dataset_orin/懂物料_augmented.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"wrote {len(records)} augmented item-grounding examples to {out_path}")
from collections import Counter
print(Counter(r[0]["response"].split("|")[0][2:] for r in records))
