# OneReason-0.8B 强化学习(RL)训练方案设计

> 纯设计文档，不含已跑实验。目标：撬动懂推荐(占总分50%)——报告证明 RL 给 +12%~73% on Recall@K，是 SFT 到顶(~0.86)后唯一的实质增长路径。
> 配套阅读：`PROGRESS.md`（SFT 阶段的所有教训与铁律）、OneReason 技术报告 §6。

---

## 0. 所有 RL 方案的共同前提与关键决策

### 0.1 目标指标
懂推荐(R3) = **Pass@K / Recall@K**：采样一组候选 itemic token → 解码成 item ID → 看真实物品命中/覆盖。**候选多样性是命根子**（报告 reward 里专门有 diversity 项）。

### 0.2 三个必须先想清楚的约束

**① reward 只能是 proxy（重要）**
真实评测把 itemic token 解码成 item ID、在 item 粒度算召回。**我们没有 itemic tokenizer / item catalog，无法解码**。所以本地 reward 只能在 **itemic-token 字符串层面**算（生成的 `<|domain_begin|><s_a><s_b><s_c>` 是否精确等于 ground-truth 的那串）。这是 proxy，和真实指标高度相关但不完全等同。→ RL 训完仍需靠正式评测确认。

**② thinking vs non-thinking：RL 阶段可能要反过来**
- SFT 阶段：non-thinking > thinking（已证实，champion 就是 non-thinking）
- **RL 阶段：报告显示 RL 后 thinking > non-thinking**（"specialize-then-unify" 解锁 thinking 收益）
- 报告的 GRPO rollout 是"N条CoT × 每条K个itemic序列"，**本身带 thinking**
- **决策**：RL 大概率要用 thinking 模式的起点 checkpoint。我们的 champion 是 non-thinking-only，可能需要：先训一个 thinking-capable 的 SFT checkpoint（保留CoT训练，如之前的 raw）作为 RL 起点；或做 non-thinking GRPO 先验证 pipeline。**这是要先定的分叉。**

**③ itemic token 熵塌缩红线（延续SFT铁律）**
报告 Figure 12a 明示：对 itemic token 做太激进的策略更新会 **entropy collapse**，杀死召回多样性（我们 exp2 就是这么崩的）。所以所有 RL 方案里，**itemic token 的更新必须"温柔"**（tight clip / 低有效步长 / diversity 奖励），这是硬约束。

### 0.3 共享基础设施（所有方案都要先搭这些）
1. **Reward 函数**：给定 prompt + 生成的 itemic token，判断是否命中 ground-truth 集合（rule-based，返回0/1）；再叠加 diversity 奖励（同一reasoning下K个候选的首子token去重数）。
2. **Rollout 生成**：从当前 policy 采样。报告的"两阶段 rollout"（N条CoT × K个itemic）能用少量CoT摊薄reward成本，是关键效率技巧。
3. **训练 prompt 来源**：官方懂推荐数据（19k），或从50万用户 UserProfile 现造真实 context（后者量大、更贴近评测分布）。
4. **框架**：GRPO 用 **TRL 的 GRPOTrainer**（最易上手，支持自定义 reward + LoRA）或 verl/OpenRLHF（更工业但重）；RFT/DPO 可直接用 LLaMA-Factory。

### 0.4 LoRA vs 全参（延续SFT结论）
- **一律先 LoRA**：显存可行、提交友好、结构上冻结itemic层→对熵更安全。
- 全参RL上限更高但吃显存、易塌缩，作为A100后备。

---

## 0.5 探测结果（2026-07-06，用 exp4 策略在官方懂推荐 prompt 上 rollout）

跑 `rl/rollout.py`（400 prompt，N=32）实测命中/信号密度：

| 域 | n | exact命中率 | **分级信号密度** | avgGraded |
|---|---|---|---|---|
| ad | 42 | 2.4% | 28.6% | 0.117 |
| living | 20 | 0.0% | 10.0% | 0.030 |
| prod | 14 | 0.0% | 14.3% | 0.043 |
| video | 324 | 0.6% | 33.3% | 0.115 |
| **全部** | 400 | **0.8%** | **31.0%** | 0.109 |

**两个决定性结论：**
1. **exact 命中率仅 0.8%**（与报告 SFT 模型 Video Pass@64≈1% 同量级）→ **RFT 不可行**：挖不到足够成功轨迹（每~6400次生成才1条）。
2. **分级部分奖励把信号密度从 0.8% 拉到 31%（约40倍）** → **GRPO 变得可行**。模型很难精确生成对物品，但**经常能生成对品类（s_a）**，分级奖励把这个"方向对了"的信号利用起来，天然形成"先对品类、再refine"的课程。

**→ 路线修正：跳过 RFT，直接 GRPO + 分级奖励。** ⚠️ 分级信号主要是品类(s_a)命中，能否转成真实 item 级 Recall@K 提升，仍需 GRPO 训完正式评测确认（proxy 老问题）。

---

## 方案 A：RFT（Rejection Sampling Fine-tuning）—— ❌ 已降级：exact命中太稀疏，不可行

**核心思想**：不搞在线RL循环。用当前模型对每个prompt采样多个候选 → **只保留命中ground-truth的那些"成功轨迹"** → 把这些成功轨迹当新SFT数据再训一遍。本质是"自己给自己造高质量数据"。

**为我们的任务怎么落地**：
1. 拿 SFT checkpoint（champion 或 thinking版），对每个懂推荐prompt采样 N=32~64 个候选
2. reward函数筛出命中的（itemic token精确匹配ground-truth）
3. 命中的 (prompt, CoT, itemic) 三元组 → 组成 RFT 数据集
4. 用 LLaMA-Factory 普通 SFT 在这批数据上再训（LoRA）

**优点**：
- 最简单，**没有在线RL的不稳定性**，就是"采样+过滤+SFT"
- **当前24GB卡就能做**（采样 + 普通SFT）
- 报告证明 RFT 在**大K时**(Recall@8/32/64) 明显提升召回覆盖——正对懂推荐

**缺点**：
- 受限于当前模型能采到的成功样本（命中率低时成功样本少）
- 提升幅度不如完整GRPO

**建议**：**RL的第一步就做RFT**，成本最低、能验证reward pipeline、且报告显示它对Recall@K有效。

---

## 方案 B：GRPO（recommendation-oriented）—— 报告主方法，核心

**核心思想**：在线RL。对每个prompt采样一组rollout，用**组内相对优势**（谁比组内平均好）更新策略，让高reward的轨迹概率上升。

**为我们的任务怎么落地**（报告§6完整配方）：
1. **两阶段 rollout**（eq6）：对每个用户，先采 N 条 CoT，每条CoT再并行采 K 个 itemic 序列 → N×K 个候选，只算 N 次推理成本
2. **混合 reward**（eq7-9）：`R = R_accuracy × R_diversity`
   - `R_accuracy`：itemic token 是否命中 ground-truth 集合（rule）
   - `R_diversity`：同一CoT下K个候选里，首子token(s_a)有多少种不同值 → 奖励覆盖不同品类
3. **组相对优势**（eq5）：`Â = (R - mean) / (std + δ)`
4. **两个稳定器（关键，防熵塌缩）**：
   - **Stage-wise clipping**（eq11-12）：CoT token 用松 clip(0.2/0.28)，**itemic token 用紧 clip(0.1/0.15)** → 防itemic分布塌缩
   - **负样本降权**（eq13）：命中率低→大量miss样本会主导梯度，给非命中rollout降权(β<1)

**优点**：
- **报告的主力方法**，+12-73% on Recall@K 主要来自它
- diversity奖励直接对齐"召回要多样候选"的目标

**缺点**：
- 在线RL，实现复杂（rollout+reward+优势+两个稳定器）
- 显存/算力吃紧，**建议A100**
- 命中率低(~1-5%)→需要够大的rollout和负样本降权才有有效信号

**框架**：TRL GRPOTrainer + 自定义 reward + LoRA。stage-wise clipping / 负样本降权可能要改trainer源码（TRL默认单clip）。

---

## 方案 C：DPO（偏好学习）—— GRPO 的轻量替代

**核心思想**：不用reward模型/在线采样。构造**偏好对**(chosen=命中的候选, rejected=没命中的候选)，用DPO loss直接拉开两者概率差。

**为我们的任务怎么落地**：
1. 采样候选，reward筛出命中(chosen)和未命中(rejected)
2. 组成 (prompt, chosen, rejected) 偏好对
3. LLaMA-Factory 直接支持 DPO（`stage: dpo`），LoRA

**优点**：
- 比GRPO简单、稳定（离线、无优势估计）
- **LLaMA-Factory原生支持**，当前卡可做
- 有现成成功/失败样本就能构造

**缺点**：
- 偏好学习是"两两拉开"，**不直接优化集合级Recall@K**，也没有diversity项 → 对多候选召回的对齐不如GRPO
- 有把分布训尖的风险（DPO会拉高chosen概率）→ **需要监控 div_sa 护栏防塌缩**

**建议**：作为"RFT之后、GRPO之前"的中间选项，或GRPO工程量太大时的替代。

---

## 方案 D：MOPD / 多teacher蒸馏 —— 进阶，最后做

**核心思想**（报告§6.3）："specialize-then-unify"：先对每个域(视频/广告/电商/直播)单独RL出4个domain专家，再把它们蒸馏进一个统一student。

**为我们的任务**：工程量最大（要先有4个域专家 + on-policy蒸馏 + information-gain过滤）。**竞赛初赛大概率用不到**，除非到了"单模型已到顶、要压榨跨域平衡"的阶段。列在这里做完整性，短期不做。

---

## 推荐路线图（按投入产出排序）

| 阶段 | 方案 | 硬件 | 状态 |
|---|---|---|---|
| RL-0 | 共享基础设施（reward + rollout + prompt集） | 当前卡 | ✅ 已完成，探测证实分级奖励可行 |
| ~~RL-1~~ | ~~RFT~~ | — | ❌ 放弃：exact命中0.8%，挖不到数据 |
| **RL-2** | **GRPO + 分级奖励**（报告主方法，diversity+两稳定器） | **A100** | 🎯 主攻。reward已就绪(`accuracy="graded"`)；训练脚本 `rl/train_grpo.py` 框架已搭 |
| RL-3 | DPO（GRPO工程受阻的替代） | 当前卡 | 备选 |
| RL-4 | MOPD多teacher蒸馏 | A100多卡 | 复赛后期 |

**关键先决问题（开工前要定）**：
1. RL 用 thinking 还是 non-thinking 起点？→ 建议先准备 thinking-capable SFT checkpoint 当起点（报告 RL 收益在 thinking 模式）。探测用的 exp4 是 non-thinking，若走 thinking 起点需另训。
2. Prompt 用官方19k还是从50万UserProfile现造？→ 混用：官方保format/质量，UserProfile扩量贴近真实分布。
3. reward 是 proxy（分级信号主要是品类s_a命中），训完必须正式评测确认。

---

## 已完成（RL-0，✅）
- [x] `rl/reward.py`：itemic命中(exact) + **分级部分奖励(graded, 0.3/0.6/1.0)** + diversity + TRL接口。CPU自测通过。
- [x] `rl/rollout.py`：采样 + 分域统计(exact命中率 vs 分级信号密度)。探测证实分级奖励把信号从0.8%→31%。
- [x] `rl/build_rl_prompts.py`：从官方懂推荐构造 (prompt, gold) 集。
- [x] `rl/train_grpo.py`：GRPO 训练脚本框架（TRL GRPOTrainer + LoRA + 分级reward），等A100直接跑。

## 24GB 卡验证结论（2026-07-06）
- ✅ 管线跑通：修好 TRL0.24+transformers5.6 兼容（`pip install mergekit llm-blender weave` + `TRANSFORMERS_CACHE`/`warnings_issued` shim，均写进 `train_grpo.py`）。
- ✅ 显存：G=16 / prompt 1536 ≈ 18.9GB（G=16 / prompt 2048 ≈ 23.9GB 快 OOM，故降到 1536）。
- ⚠️ reward 训练时一直平在 ~0.005 没上升 → 200-300 步、无稳定器、beta=0、**prompt 截到 1536** 都是原因。
- ⚠️ **prompt 长度隐患**：`build_rl_prompts.py` 保留到 4096，但 GRPO 训练截到 1536 → 懂推荐长用户历史被左截断切掉，上下文变少，信号可能弱于探测时的 31%（探测用完整 prompt）。

## 待做（RL-2，等A100）
- [ ] **prompt 长度放开到 2048–4096**（当前 1536 是 24GB 卡的妥协）。A100(40/80GB)让模型看到完整用户历史 → 信号更强，是 reward 能否上升的关键之一。
- [ ] 在 `train_grpo.py` 里实现 stage-wise clipping（itemic紧clip防塌缩）+ 负样本降权（需子类化 GRPOTrainer）
- [ ] 放开 G（24GB 只到 16；A100 可 32+）、更多步（当前 200-300 太短）
- [ ] 决定 thinking/non-thinking 起点，跑 GRPO，div_sa 护栏 + 正式评测确认
- [ ] 从 UserProfile 扩充 RL prompt 集（当前只用官方懂推荐 2000 条）
