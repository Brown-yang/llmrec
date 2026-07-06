# LLaMA-Factory 上游版本

本目录是从 LLaMA-Factory 官方仓库 clone 后使用的快照（已去掉嵌套 .git 以便纳入本项目仓库）。

- 上游仓库：https://github.com/hiyouga/LLaMA-Factory.git
- clone 时的 commit：`a48af5cc690aa1f3a452526f5e30ec58b8c08eaa`
- 版本：0.9.6.dev0

如需更新框架，可到上游仓库重新 clone。本项目对框架源码未做修改，仅新增了：
- `examples/train_lora/onereason_*.yaml`、`examples/train_full/onereason_*.yaml`、`examples/merge_lora/onereason_*.yaml`（训练/合并配置，另在项目根 `configs/` 也有备份）
- `data/dataset_info.json`（注册了 onerec_sft* 等数据集）

不纳入 git 的大文件（见 .gitignore）：`saves/`（训练产出的模型/checkpoint）、`data/onerec_sft*.jsonl`（生成的训练数据）。
