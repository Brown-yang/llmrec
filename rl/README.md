# RL 基础设施(RL-0)

配套 `../RL_DESIGN.md`。这里是所有 RL 方案(RFT/GRPO/DPO)共享的、**不占卡就能准备好**的组件。

## 文件

| 文件 | 作用 | 状态 |
|---|---|---|
| `reward.py` | 报告 §6 的 reward:itemic命中(R_rule) × 多样性(R_div)。含 TRL-GRPOTrainer 兼容的 `make_grpo_reward_func`，以及 RFT/DPO 用的 `group_reward`/`itemic_hit` | ✅ CPU自测通过 |
| `build_rl_prompts.py` | 从官方懂推荐数据抽 (prompt, gold_items) 作为 RL prompt 集 | ✅ 已验证 |
| `rollout.py` | 采样候选 + reward 打分;`--dump-hits` 直接产出 RFT 数据 | ⏳ 需GPU，reward逻辑已CPU测 |

## ⚠️ 核心前提(务必先读 RL_DESIGN.md §0)

- **reward 是 proxy**:无 itemic tokenizer，只能在 itemic-token 字符串层面判命中。训完必须正式评测确认。
- **thinking/non-thinking**:SFT 阶段 non-thinking 好，但 **RL 后 thinking 更好**。prompt 集默认 `nothink`（匹配 champion 起点）；有了 thinking-capable 起点后用 `--mode think`。
- **itemic 熵塌缩红线**:RL 更新 itemic token 必须"温柔"(GRPO 的 tight-clip / diversity 奖励)，否则重蹈 exp2。

## 用法示例

```bash
# 1. 构造 RL prompt 集(官方懂推荐)
python build_rl_prompts.py --out ../datasets/rl_prompts_tuijian.jsonl

# 2. RFT 第一步:用当前最优 checkpoint 采样，把命中的轨迹导出成 SFT 数据
python rollout.py --adapter ../LLaMA-Factory/saves/onereason-0.8b/lora/exp4 \
    --prompts ../datasets/rl_prompts_tuijian.jsonl \
    --n 32 --limit 2000 --dump-hits ../datasets/rft_hits.jsonl
#   -> 打印 Pass@32 / 命中率 / 多样性；产出 rft_hits.jsonl
#   -> 把 rft_hits.jsonl 注册进 dataset_info.json，用 LLaMA-Factory 普通 SFT 再训(LoRA) = RFT

# 3. GRPO(A100):用 reward.make_grpo_reward_func() 作为 TRL GRPOTrainer 的 reward_func
#    (需在 GRPOTrainer 上加 stage-wise clipping + 负样本降权，见 RL_DESIGN.md 方案B)
```

## 下一步(RL-1，等卡)
- 先跑 `rollout.py` 看 Pass@N 命中率:决定 reward 信号够不够强、要不要更大 N
- 命中率可用 → 做 RFT(方案A)；RFT 有效 → 上 GRPO(方案B，A100)
