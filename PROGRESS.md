# OneReason-0.8B 竞赛微调 · 工作进展与交接文档

> 最后更新：2026-07-05（读完 OneReason 技术报告全文，重大策略修正）
> 用途：交接给 agent 继续工作。读完本文即可知道「做过什么、结论是什么、从哪继续」。

---

## 0. 一句话现状

当前竞赛最高分仍是 **v1.0.0 = 0.8596**（官方数据 + 1 epoch + **纯 LoRA + non-thinking**，什么都没改）。之前的「改进」都没超过它：exp1=0.8529、exp4=0.8063、raw=0.7575、v1.0.2=0.7354、v1.0.3=0.7212——因为都在动 champion 配方的**正确部分**（增强/thinking/解冻itemic/智能截断），越动越差。

**转折（2026-07-07）：从评测 log 逆向分析发现 champion 有两个"权重-数据错配"缺口**——**懂用户**(25%权重却只3.6%数据，长样本被4096截断丢了61%)、**懂世界**(12.5%权重却0官方数据，全靠基座)。→ **exp5** 首次针对性补齐：cutoff 8192 恢复懂用户(1120→2869) + 从CMMLU/GSM8K/LogiQA构造懂世界(0→5000) + 保懂推荐51%不稀释。**exp5 已训完(eval_loss 1.201, div_sa 17.12健康, adapter可提交), 待正式评测——这是逻辑最强、最可能超 champion 的一次(补缺口而非动正确部分)。**

**长期：SFT 补完缺口后天花板仍在 ~0.86-0.88，冲 0.9 靠 RL(GRPO)** 撬动占 50% 权重的懂推荐。RL 基础设施已搭好（`RL_DESIGN.md`/`rl/`，分级奖励验证 31% 信号密度，24GB 卡跑通 GRPO），等 A100 补两个稳定器开跑。冲 0.9 两点见 §9。

---

## 1. 竞赛规则要点

- **赛事**：快手探索者 LLM-Rec 挑战赛，平台 = 快手万擎（StreamLake）<https://www.streamlake.com/product/wanqing>
- **基座模型**：`OneReason-0.8B-pretrain-competition`（本地路径 `/home/lab/wy/LLM_REC/OneReason-0.8B-pretrain-competition`）
  - Qwen3ForCausalLM，28 层，hidden 1024，**vocab_size 176253**（含扩展的 itemic token）
  - 初赛只能基于此模型迭代，**评测严格校验 config 与 baseline 一致**，不能改架构
- **评测**：OneRec Benchmark，四维度 `懂物料 / 懂用户 / 懂推荐 / 懂世界`
  - **总分 = 8 个子分数直接相加**（不是平均）：懂物料×1 + 懂用户×2 + 懂推荐×4 + 懂世界×1
  - ⚠️ **懂推荐拆成 4 个子任务，占 8 个槽位里的 4 个 = 隐性权重 50%**，是决定总分的大头
  - **正式评测每天限 3 次**，评测本身耗时 40-50 分钟 → 必须靠本地 proxy 调参，正式评测只留给有把握的版本
- **提交方式**（万擎官方入口，均支持）：
  - **LoRA**：上传 `adapter_model.safetensors` + `adapter_config.json`（**无需 merge 成完整模型**）
  - 全参：上传 `model.safetensors`（分片则加 `model.safetensors.index.json`）
  - 训练方法选 lora/全参，模型类型选「文本生成」
- 线下训练官方建议 Transformers **v5.3.0**（⚠️ 我们环境是 5.6.0，见 §2 风险）
- 允许外部数据、蒸馏；不鼓励模型融合；复赛需交数据构造脚本+训练脚本复现

---

## 2. 环境

- **conda env `onerec`**（python 3.11）：`source /home/lab/miniconda3/etc/profile.d/conda.sh && conda activate onerec`
- torch **2.11.0+cu128**（匹配驱动 CUDA 12.8），torchvision/torchaudio 也是 cu128 版
- **LLaMA-Factory 0.9.6.dev0**，路径 `/home/lab/wy/LLM_REC/LLaMA-Factory`，transformers **5.6.0**
- **GPU：单卡 RTX A5000 24GB**（桌面占用约 5GB，可用约 19GB）。⚠️ **用户 A100 服务器即将就绪，之后按 A100 来**
- 训练统一加 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解显存碎片
- ⚠️ **风险**：transformers 5.6.0 vs 官方建议 5.3.0。目前无报错，但若评测端严格校验版本兼容，可能是隐藏风险，必要时固定到 5.3.0 重装。

---

## 3. 数据

### 3.1 官方 SFT 数据（万擎平台下载）
路径：`/home/lab/wy/LLM_REC/datasets/dataset_orin/`，共 **32,480 条**，格式每行 `[{"system","prompt","response"}]`
- 懂推荐1-4.jsonl：**19,204**（权重最大）
- 懂物料part1-7.jsonl：**10,384**（part1商品/part2主播/part3广告/part4短视频=描述→token；part5商品/part6广告/part7短视频=token→描述）
- 懂用户.jsonl：**2,892**（长行为时间线→按主题筛选相关行为，JSON 数组输出）

### 3.2 HF 原始行为大数据集
路径：`/home/lab/wy/LLM_REC/datasets/OpenOneRec/Explorer_LLM_Rec_Competition/data/`
- `OneReason_UserProfile/`：**50 万用户**原始多域行为序列（raw pid，非 SID），63 字段
- `OneReason_Pid2Sid/`：pid → 三段式语义ID `[s_a,s_b,s_c]`（198 parquet）
- `OneReason_Pid2Caption/`：pid → 文字描述（136 parquet，video是段落/goods是短标题/live是关键词列表）
- `OneReason_Pid2Tag/`：pid → 三级类目标签
- `OneReason_General/`：通用对话/推理数据（保通用能力用）
- join 规则：`(domain, pid)`，domain 取值 `video/video`(视频) `video/ad`(广告) `goods`(电商) `live`(直播)

### 3.3 自建增强数据（写在 dataset_orin/ 下）
- `懂物料_augmented.jsonl`：**26,522** — 从官方数据揪出「没 grounding 的目标物品」→ Pid2Sid 反查 pid → Pid2Caption 拿真实描述 → 套官方模板生成。**这份质量可信。**
- `懂推荐_augmented.jsonl`：**25,000** — 从 UserProfile 抽 2.5万用户，按域切「历史→目标」构造。⚠️ **有 bug（见 §6）**
- `懂世界_augmented.jsonl`：**2,500** — OneReason_General 采样，保通用能力

---

## 4. 数据处理管线

主脚本：`/home/lab/wy/LLM_REC/datasets/prepare_sft_data.py`
处理步骤：①展平数组 ②剥离 response 开头的 `<think>...</think>`（做 non-thinking SFT）③统一 prompt 结尾为 `/no_think` ④超长样本**智能截断**（保留结尾指令，二分查找裁掉最早期行为历史，插入「（注：更早期的行为记录已省略）」提示，而非整条丢弃）
- 环境变量：`EXCLUDE_AUGMENTED=1` 只用官方数据；`OUT_TRAIN=<path>` 指定输出
- 输出注册在 `LLaMA-Factory/data/dataset_info.json`：`onerec_sft`（全量86k）、`onerec_sft_exp1`（官方32k）

增强数据构造脚本（都在 datasets/）：
- `build_item_grounding_augment.py` → 懂物料_augmented（依赖 /tmp/found_captions.pkl）
- `build_rec_augment.py` → 懂推荐_augmented ⚠️**有 goods bug**
- `build_general_augment.py` → 懂世界_augmented

---

## 5. 本地评估工具（附带一个重要教训：本地 proxy 的 Pass@K 不可信）

**固定评估集**：`/home/lab/wy/LLM_REC/datasets/eval_set.jsonl`（900 条，物料/用户/推荐各 300，**只从官方数据抽、永不参与训练、每次实验都用同一批**）。构建脚本 `build_eval_set.py`。
- ⚠️ **局限**：目标物品是"训练分布内"的，且每条只有 **1 个 gold**。而真实竞赛测的是 **Recall@64 over 平均 14 个"全新"目标物品**（报告 Table 22：Video 95.3% 训练没见过）。**分布不匹配 → 本地命中率会奖励记忆/塌缩，与真实分数反相关。**

**两个评估脚本**：
1. `eval_split_loss.py` —— teacher-forced `itemic_loss` + 贪婪 `Pass@1`。**❌ 已废弃**：与真实 Recall@K 反相关（越自信=分布越尖=多样性越差=真实召回越低）。当初就是它误导我们提交了 exp2。
2. `eval_split_loss_recall.py` —— 采样式 `Pass@k/Recall@k` + **多样性 `div_sa/div_tok`**。用法 `python eval_split_loss_recall.py --adapter <dir> --k 32 --limit 300 --categories tuijian`。

**⚠️ 回溯验证结论（2026-07-05，两个 proxy 都做了）**：拿"已知 exp2<champion"这个事实去验证，结果——
- **两个 proxy 的 Pass@K 都反了**：exp2 本地 Pass@32=4.7% > exp1 的 1.3%，但真实 exp2 更差。**根因**：本地评估集是"分布内+单目标"，exp2 熵塌缩后概率集中到少数"像训练数据"的高频物品→本地命中虚高，但真实"新物品+多目标"召回需要多样性→真降。
- **唯一可靠的信号是 `div_sa`（多样性）**：exp2=15.42 < exp1=17.40，**正确检测到熵塌缩**。→ 把 `div_sa` 当**"塌缩护栏"**：训练后测一下，多样性掉了就是危险信号，别提交。

**结论**：**无法用现有数据造出可信的分数 proxy**（不知道竞赛测试集的新物品分布、造不出多目标 gold）。务实三条腿：①靠报告的原则性设计（non-thinking / 冻结 itemic / CoT 混合 / RL 带多样性奖励）②`div_sa` 当护栏 ③正式评测省着用，只留给"设计上确实不同"的方案。**别再靠本地 Pass@K 微调调参。**

---

## 6. ⚠️ 已知 Bug（待修）

**`build_rec_augment.py` 的 goods 域历史字段用错了**：
- 用了 `ec_colossus_rs_item_id_list` 当「浏览了商品」，但实测这字段是「系统曾展示的候选」，**86.8% 从未被点击**
- 等于给模型喂了大量假的「浏览行为」噪音 → 拖累 v1.0.2 得分
- **修法**：改用 `ec_good_click_item_id_list_extend`（真实点击，中位320）或 `ec_good_order_item_id_list_extend`（购买）
- 这是 v1.0.2 得分暴跌（0.7354）的主因之一

---

## 7. 实验记录

### 7.1 正式评测（消耗每日 3 次配额）

| 版本/模型 | 配置 | 总分 | 懂物料 | 懂用户(2项) | 懂推荐(4项) | 懂世界 | 备注 |
|---|---|---|---|---|---|---|---|
| **v1.0.0** | 官方32k, 1ep, **纯LoRA, non-thinking** | **0.8596** | 0.1533 | 0.1006 | **0.4726** | 0.1331 | **CHAMPION** |
| v1.0.1 | 官方32k, 3ep, LoRA | 0.8447 | 0.1533 | 0.1194 | 0.4323 | 0.1398 | 3轮过拟合 |
| exp1 | 官方32k, 1ep, LoRA, **智能截断**, non-thinking | 0.8529 | **0.1840** | 0.1049 | 0.4373 | 0.1268 | ≈champion；懂物料↑但懂推荐↓(智能截断伤懂推荐) |
| v1.0.2 | 增强86k, ~1000step, LoRA | 0.7354 | 0.1533 | 0.0911 | 0.3437 | 0.1472 | goods bug + 未训完 |
| v1.0.3 | exp2=官方32k+**解冻embed/lm_head**(全参提交) | 0.7212 | 0.1533 | 0.1053 | 0.3310 | 0.1316 | 熵塌缩,懂推荐崩 |
| **raw@1500** | 官方32k, **最少处理:保留CoT+/think(thinking模式)**, LoRA | **0.7575** | 0.1533 | 0.0947 | **0.3987** | 0.1108 | **近收敛(eval1.436≈final1.429)。thinking模式伤懂推荐** |
| **exp4** | 官方32k + **懂物料grounding增强26.5k**, 1ep, LoRA, drop截断, non-thinking | **0.8063** | **0.1226** | 0.0935 | 0.4600 | 0.1301 | **懂物料增强backfire**：全维度都低于champion，懂物料掉最多(0.1226<champion0.1533<exp1 0.184) |
| **exp5** | **cutoff 8192(恢复懂用户)+ 懂世界5000(新)**, 官方懂推荐/物料全量, 1ep, LoRA, non-thinking | **待评测** | — | — | — | — | **修3个数据缺口**：懂用户1120→2869、懂世界0→5000、懂推荐保51%不稀释。div_sa 17.12(健康)。**最有希望超champion的实验** |

**两条被硬证实的规律**：
1. **thinking < non-thinking(报告Table 14 + raw实测双重证实）**：raw 保留 CoT+`/think` 走 thinking 模式，近收敛也只有 0.7575，懂推荐 0.3987 全场最低。**champion/exp1 高，核心是 non-thinking**。报告说的"CoT有用"是"训练带CoT + 推理non-thinking"，raw 是"训练带CoT + 推理thinking"，组合错了。
2. **"处理越少越好"被反驳**：exp1(**完整处理**,non-thinking)=0.8529≈champion；raw(**最少处理**,thinking)=0.7575。**处理动作本身无害**——剥CoT+强制no_think 恰好把模型推到正确的 non-thinking 模式，是帮忙不是添乱。真正的变量是 non-thinking vs thinking，不是处理多少。
3. 分数下降元凶归因：v1.0.1=3轮过拟合；v1.0.2=goods bug；v1.0.3=熵塌缩(embed/lm_head)；raw=thinking模式；**exp4=懂物料增强backfire**。**champion 是纯 LoRA + non-thinking，问题从来不是 LoRA、不是"处理"。**
4. **⚠️ 数据增强全部 backfire（2026-07-06，exp4 log 诊断）**：exp4 加了 26.5k 懂物料 grounding 增强，懂物料反而从 champion 的 0.1533 掉到 **0.1226**（exp1 无增强 0.184 最高）。log 量化分析：三者候选多样性都~10(distinct s_a)，**不是多样性问题，是准确性**——自建 caption→token 映射和官方评测的精确映射有偏差，训练后把 grounding 带偏。**教训：本地 div_sa 护栏只测懂推荐、测不到懂物料任务，对这次盲；规则/自建增强数据质量不够，SFT 阶段一律别加。**
5. **SFT 已到天花板（横向铁证）**：champion 0.8596 是所有提交里最高，exp1/exp4/raw/v1.0.2/v1.0.3 **无一例外全部更低**。每一个对 champion 配方的改动（增强/展平/thinking/解冻itemic/智能截断）都降分。**SFT 层面 ~0.86 就是上限，突破要靠 RL。**

### 7.2 本地对照实验（不消耗评测配额，用固定评估集，limit 300）

| 实验 | 配置 | eval_loss(训练) | 懂物料 itemic/P@1 | 懂用户 itemic/P@1 | 懂推荐 itemic/P@1 | ALL itemic/P@1 |
|---|---|---|---|---|---|---|
| base | 原始未微调 | — | 7.66 / 0% | 2.05 / 0% | 3.77 / 0% | 2.75 / 0% |
| **exp1** | 官方32k+智能截断, 1ep, LoRA all | 1.349 | 2.81 / 1.0% | 0.61 / 0% | 3.54 / 1.1% | 1.26 / 0.8% |
| **exp2** | exp1 + 解冻 embed/lm_head | 1.424 | **2.32 / 3.0%** | **0.43 / 2.0%** | **3.31 / 1.1%** | **1.05 / 2.0%** |

- exp1 配置：`LLaMA-Factory/examples/train_lora/onereason_lora_sft_exp1.yaml`，输出 `saves/onereason-0.8b/lora/exp1/`
- exp2 配置：`LLaMA-Factory/examples/train_lora/onereason_lora_sft_exp2.yaml`（唯一区别 `additional_target: embed_tokens,lm_head`），输出 `saves/onereason-0.8b/lora/exp2/`
- **注意**：`additional_target` 在 LLaMA-Factory 里是**全量训练**这两层（不是低秩），exp2 可训练参数 366M(31%)。单卡训练时显存峰值约 23.4GB/24.5GB（能跑但很紧）。
- ⚠️ 训练 eval_loss（从训练数据切2%）与固定评估集方向不一致：训练eval_loss exp2更高，但固定均衡评估集 exp2 全面更好。**以固定评估集为准**（同 900 条、均衡、可比）。

### 7.3 exp5：权重-数据错配诊断 + 数据缺口补齐（2026-07-07，从评测log分析得出）

**从 champion(v1.0.0) log 逆向分析出的3个数据缺口：**

| 维度 | 评测权重 | champion数据占比 | 问题 |
|---|---|---|---|
| 懂推荐 | 50% | 62.5%(19204) | ✓ 匹配 |
| 懂物料 | 12.5% | 33.8%(10384) | 数据过剩 |
| **懂用户** | **25%** | **3.6%(1120)** | **严重欠训练**：官方2892条，中位数4487 token>4096，被drop-long砍掉1772条(61%) |
| **懂世界** | **12.5%** | **0%** | **官方零数据**：得分~0.13纯来自基座预训练 |

**懂世界数据从哪来（官方完全没有）：** 官方 `Explorer_LLM_Rec_Competition` 只有物品(Pid2*)/用户(UserProfile)/通用(General=英文编程翻译, 非常识MCQ)，**无懂世界**。评测懂世界=中文四选一(常识+数学+逻辑，见log)。→ 用 `datasets/build_shijie.py` 从3个外部中文MCQ数据集构造，裸字母答案+/no_think对齐评测格式：
- **CMMLU**(`haonan-li/cmmlu`, HF)→ 常识/学科
- **GSM8K_zh**(`meta-math/GSM8K_zh`, hf-mirror, 自由问答→转MCQ加数字干扰项)→ 数学
- **LogiQA**(`lgw863/LogiQA-dataset`, GitHub, zh_train)→ 逻辑推理
- ⚠️ 英文MMLU弃用（语言不匹配中文评测，exp4式风险）

**exp5 数据构成(37457条, cutoff 8192)：** 懂推荐19204(51%) + 懂物料10384(28%) + 懂用户2869(8%, cutoff恢复) + 懂世界5000(13%, 2000常识+1500数学+1500逻辑)。**核心：保懂推荐≥50%不稀释(exp4教训) + 补齐两个欠账维度**。
- 配置 `examples/train_lora/onereason_lora_exp5.yaml`；数据 `prepare_sft_data.py` 加了 `MAX_TOKENS` env(设8192)。
- cutoff 8192 显存：LoRA+梯度检查点，24GB峰值~22.9GB(紧但稳，4.25h无OOM)。
- final eval_loss 1.201，div_sa 17.12(健康，懂推荐没塌/没稀释)。**懂用户/懂世界的增益 div_sa 测不到，待正式评测确认。**
- ⚠️ **loss跨实验不可比**：exp5 eval_loss 1.20 < exp4 1.41，但因懂世界MCQ短答案拉低平均，非质量指标。

---

## 8. 核心发现（读完技术报告后的最终诊断）

> ⚠️ 本节已根据 OneReason 技术报告（arXiv:2606.06260）全文重写。之前"解冻 embed/lm_head 有效"的结论**是错的**——那是被错误 proxy 误导。

### 8.1 之前 proxy 为什么把我们带偏了（最重要的教训）
- 我们的 `eval_split_loss.py` 测的是 **teacher-forced itemic_loss + 贪婪 Pass@1**。
- 但报告 §3.2 / 附录 B.4 明确：**懂推荐(R3)评测 = Pass@K/Recall@K**——采样一组候选、解码成 item、看召回，**候选多样性是命根子**（RL 奖励里专门有 `R_div` 奖励第一位 sub-token 的多样性，eq 7-9）。
- **teacher-forced itemic_loss 和 Recall@K 反相关**：交叉熵越低=分布越尖=采样多样性越差=Recall@K 越低。所以 exp2 在旧 proxy 上"更好"，正式评测反而崩。

### 8.2 exp2/embed-lm_head 训练为什么是死路（报告三重印证）
1. **Figure 12(a) 直接命名 "itemic token entropy collapse"**：对 itemic token 做太激进的更新会熵塌缩，报告专门用**更紧的 clip** 保护 itemic token。exp2 全量高 LR 训 lm_head = 反着来。
2. **训练配方 Table 4**：embed/lm_head 只在**预训练 Stage 1**（110B token、冻结文本层、LR 2e-4→1e-5）训练一次让 itemic 嵌入 settle。下载的 checkpoint 已做过。在 32K SFT 数据上重训=破坏预训练学好的空间。
3. **GRPO 讨论**：连 RL 都会"sharpen the output distribution... not fully aligned with recommendation"，所以才要加多样性奖励。

### 8.3 报告确认我们对的两个大方向
- **懂推荐 SFT non-thinking > thinking**（Table 9/14，thinking 只有 RL 后才反超）→ 我们用 `/no_think` 正确。
- **纯 LoRA 天然冻结 embed/lm_head → 保住 itemic 分布熵** → 这正是 champion(LoRA)赢、exp2(全量动itemic层)崩的真正原因。

### 8.4 任务本质：主要是泛化，不是记忆（Table 22）
| 域 | 记忆型 | 泛化型 |
|---|---|---|
| Video | 4.7% | **95.3%** |
| Product | 27.8% | 72.2% |
| Ad | 75.9% | 24.1% |
| Live | 73.0% | 27.0% |
- Video/Product 目标 95%/72% 训练没见过→只能泛化，Recall 天然低；Ad/Live 记忆型多→分数天然高。exp2 懂推荐第1子任务崩到 0.0096，大概率就是泛化最难的 Video 域被熵塌缩杀死。

### 8.5 报告给出的可用杠杆
- **CoT 混合训练有用**（Table 17）：保留部分 `<think>` 数据（即使 non-thinking 推理）提升召回，各域最优 CoT 比例不同（Fig 25: Video~55% / Product~95% / Live~55% / Ad~25%）。**我们把 CoT 全剥了，可能丢了这块。**
- **推理要紧凑**（Fig 8）：interest expansion 宽度 n∈{1,3,5} 优于 10/20。
- **真正大杠杆是 RL**：报告懂推荐增益(+12%~73% on Recall@K)全来自 recommendation-oriented GRPO（两阶段 rollout + accuracy×diversity 奖励 + itemic 紧 clip 防熵塌缩 + 负样本降权）+ RFT/MOPD。**竞赛不提供 RL 代码→这是最大差异化空间。**

---

## 9. 下一步（决策与待办 · 已按技术报告修正）

### 铁律（不可再违反，都是评测次数买来的）
- **必须 non-thinking**：`/no_think` + 剥 CoT。thinking 模式伤懂推荐（报告 Table 14 + raw@1500=0.7575 双证）。
- **SFT 里永远不要训练 embed_tokens / lm_head**。纯 LoRA 自动冻结它们（正确）。exp2/v1.0.3=0.7212 熵塌缩。
- **"处理动作"本身无害**（展平/剥CoT/强制no_think 是帮忙），不要因为"少处理"就保留 CoT+thinking。
- **本地 Pass@K 不可信**（分布内+单目标，奖励塌缩）；只把 `div_sa` 当塌缩护栏用（见 §5）。

### 当前最优基线
- **champion=`saves/.../lora/sft`（0.8596）** 和 **exp1=`saves/.../lora/exp1`（0.8529）**：都是纯LoRA+non-thinking，adapter 20MB 就绪可提交。exp1 懂物料更高(0.184)但懂推荐略低(智能截断所致)。

### 下一步方向（区分两类 SFT 改动）
**关键区分（2026-07-07 修正）**：SFT 改动分两类——
- ❌ **动 champion 配方的"正确部分"**：增强/展平/thinking/解冻itemic/智能截断——**全部证伪，别再试**（exp1/exp4/raw/v1.0.2/3）。
- ✅ **补 champion 没覆盖的"数据缺口"**：懂用户欠训练(25%权重)、懂世界零数据(12.5%权重)——**这是有效的非RL方向**，exp5 正在验证。

**当前待办：**
- [ ] **提交 exp5** 看是否超 champion（补懂用户+懂世界缺口，data-fix 逻辑最强）。若超→缺口方向成立，可继续优化懂世界数据质量/配比。
- [ ] 提交基线：exp5 未超则回 champion（0.8596）。
- [ ] 懂物料(exp1 曾到0.184)的来源仍未完全搞清；懂用户恢复后能涨多少看 exp5。

### 🎯 冲 0.9 的两点明确结论（2026-07-06）
从 0.86 到 0.9 需 +0.04 绝对分，只有 RL 撬动懂推荐(50%权重)能做到。**两个必须定的点：**

**一、微调方式**
- SFT 基座：**保持 champion（纯官方 + non-thinking + 冻结itemic的LoRA），不动**。这是 RL 的起点。
- 突破 0.9 的方式 = **在 champion 之上做 GRPO 强化学习**（不是任何 SFT 方法，SFT 已证明到顶）。
  - **先 LoRA GRPO**（冻结itemic→防熵塌缩，A100可行，安全）。必须补两个稳定器：stage-wise clipping（itemic紧clip）+ 负样本降权。
  - 全参 GRPO 作为更高天花板的后备（A100 80GB + itemic塌缩防护），LoRA 到顶再上。
  - **关键分叉 thinking vs non-thinking**：报告证明 RL 后 thinking>non-thinking。冲 0.9 很可能需要 **thinking-capable 的 SFT 基座 + thinking 模式 GRPO**（报告完整配方）。这比 non-thinking GRPO 赌注大但天花板高。稳妥路径：先 non-thinking GRPO 验证增益，再考虑 thinking 版。
- reward 用分级部分奖励（`rl/reward.py`，已验证 31% 信号密度；exact 命中仅 0.8% 太稀疏）。

**二、数据集**
- **SFT 数据：纯官方，绝不增强**（exp4 铁证：增强全 backfire）。
- **RL 数据（真正的新杠杆）**：RL 用的是 prompt（不是答案对），模型自己 rollout。
  - 官方懂推荐 19k prompt（`rl/build_rl_prompts.py` 已生成）保 format/质量。
  - **从 50 万 UserProfile 扩量**当 rollout 输入，用真实未来行为当 ground-truth → 直击懂推荐的泛化/覆盖问题（Table 22：95% 是训练没见过的新物品，SFT 教不会，RL 才能）。
  - 懂物料若要救：需和官方映射**精确对齐**的 grounding 数据（官方 Pid2Sid 精确解析、不改写 caption），否则像 exp4 一样越帮越忙。当前优先级低于懂推荐。

**一句话**：微调方式 = champion-SFT → GRPO（LoRA先行，thinking为高线）；数据集 = SFT纯官方 + RL用UserProfile扩量的懂推荐prompt。RL 基础设施见 `RL_DESIGN.md` / `rl/`。

### 已废弃的死路（别再走，都有评测证伪）
- ❌ **任何 SFT 数据增强**（=exp4=0.8063，懂物料 backfire；v1.0.2 也是）
- ❌ **SFT 阶段 thinking 模式 / 保留 CoT+`/think`**（=raw@1500=0.7575，报告也说 SFT thinking 更差。注意：RL 阶段 thinking 反而更好，别混淆）
- ❌ **"最少处理"当灵丹**（raw 最少处理反而最差；真变量是 non-thinking）
- ❌ 解冻/全量训练 embed/lm_head（=v1.0.3=0.7212，熵塌缩）
- ❌ 用 itemic_loss / 贪婪 Pass@1 当 proxy（与真实分数反相关）
- ❌ **本地 div_sa 护栏当万能**（只测懂推荐，对懂物料任务盲，没拦住 exp4 的 backfire）

---

## 10. 常用命令速查

```bash
# 激活环境
source /home/lab/miniconda3/etc/profile.d/conda.sh && conda activate onerec
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/lab/wy/LLM_REC/LLaMA-Factory

# 生成官方-only 数据集（exp1 用）
cd /home/lab/wy/LLM_REC/datasets && EXCLUDE_AUGMENTED=1 OUT_TRAIN=.../data/onerec_sft_exp1.jsonl python3 prepare_sft_data.py

# 训练
llamafactory-cli train examples/train_lora/onereason_lora_sft_exp2.yaml

# 本地分项评估（不占评测次数）
cd /home/lab/wy/LLM_REC/datasets
python3 eval_split_loss.py --adapter /home/lab/wy/LLM_REC/LLaMA-Factory/saves/onereason-0.8b/lora/exp2 --limit 300
python3 eval_split_loss.py --base-only --limit 300   # 原始模型地板参照

# 监控训练（不要用 TaskOutput 看缓存，直接读文件）
tail -f saves/onereason-0.8b/lora/<exp>/trainer_log.jsonl
```

## 11. 关键文件地图

```
/home/lab/wy/LLM_REC/
├── OneReason-0.8B-pretrain-competition/   # 基座模型（不可改config）
├── OneReason.pdf                          # 技术报告 arXiv:2606.06260
├── rec.txt                                # 万擎平台使用说明/FAQ
├── PROGRESS.md                            # 本文档
├── datasets/
│   ├── dataset_orin/                      # 官方12文件 + 3个_augmented
│   ├── OpenOneRec/Explorer_LLM_Rec_Competition/  # HF原始行为大数据集
│   ├── prepare_sft_data.py               # 主数据处理管线
│   ├── build_item_grounding_augment.py   # 懂物料增强
│   ├── build_rec_augment.py              # 懂推荐增强 ⚠️goods bug
│   ├── build_general_augment.py          # 懂世界增强
│   ├── build_eval_set.py                 # 生成固定评估集
│   ├── eval_split_loss.py                # 分项评估工具（核心）
│   └── eval_set.jsonl                    # 固定评估集900条
└── LLaMA-Factory/
    ├── data/dataset_info.json            # 数据集注册（onerec_sft, onerec_sft_exp1）
    ├── data/onerec_sft.jsonl             # 全量86k
    ├── data/onerec_sft_exp1.jsonl        # 官方32k
    ├── examples/train_lora/onereason_lora_sft_exp1.yaml
    ├── examples/train_lora/onereason_lora_sft_exp2.yaml
    └── saves/onereason-0.8b/lora/        # exp1/ exp2/ 等 checkpoint
```
