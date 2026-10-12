# 银行业监管制度与统计报表可信 RAG

面向银行业监管制度与统计报表的可信 RAG 问答系统。目标是保留可追溯证据、处理复杂表格，并在条件不足时避免无依据回答。

## Repository Scope

This public repository contains code, system design and demo data only.
Competition-provided data and confidential materials are excluded.

原项目保持冻结。本目录是独立公开副本；不包含比赛数据、题库、Gold、私有索引、答案或 trace。

## 架构与技术

文档解析 → 索引 → BM25 + Embedding → RRF 融合 → Cross-Encoder → 来源关联与文件内检索 → 证据筛选 → 条件判断 / Gate → 答案生成 → 来源引用。

LangGraph 编排流程；Haystack 维护检索组件；Sentence Transformers 提供 Embedding 和重排；DeepSeek 用于生成；确定性规则处理部分表格选择与计算。公开源码保留原 `src/bank_rag` 结构。详见 [架构](docs/architecture.md)。

公开版已迁入通用 Jev 证据判断标准，并在 LangGraph 问答入口提供可选的“规则 Gate OR Jev answer”放行。默认关闭，需安装可选依赖、设置开关和 TypeSafe 密钥。私有评测数据、题库、报告和实验封装未复制；此接入不复现私有最终评测结果。

## 快速开始

使用 Python 3.11+，推荐 3.12。在本目录执行：

```sh
python -m venv .venv
# 激活虚拟环境后：
python -m pip install -e ".[dev]"
python scripts/demo.py
python -m pytest tests
```

默认 demo 不调用 API、不加载 Embedding/CE 权重。将自建 Markdown/CSV 转成 DOCX/XLSX 后使用真实 parser、索引、BM25 和引用接口；答案仅为明确标记的抽取预览，证据存在检查不等于正式 Gate。LangGraph 做构建检查，未在离线预览中执行完整回答链路。

Demo 自建来源和查询含少量 English 字段标签，便于默认 BM25 的词边界匹配。纯连续中文查询首次 smoke 未召回，本次没有修改生产 tokenizer；本 demo 仅验证可运行性，不能用来证明自然中文检索质量。

可选真实生成：在 shell 设置 `DEEPSEEK_API_KEY` 等模板变量后运行 `python scripts/demo.py --ask`。该步骤有 API 成本，未作为本次 smoke test 执行。`.env` 不会自动加载。模型可用性由供应商决定，必要时设置 `DEEPSEEK_MODEL`。

完整混合检索需要模型下载，并配置 `RAG_RETRIEVAL_MODE=hybrid`；本次未验证全新环境安装及模型下载。

## 可选 Jev 证据判断

安装：`python -m pip install -e ".[dev,jev]"`。在 shell 自行安全设置 `TYPESAFE_API_KEY` 后，设置 `RAG_ENABLE_JEV=1`、`RAG_ORCHESTRATOR=langgraph`，再使用普通 CLI `ask` 或 API `/ask`。CLI `retrieve`、API `/retrieve`、legacy 回退和文件目录查询捷径不执行 Jev。`.env.example` 不会自动加载。

```powershell
$env:RAG_ENABLE_JEV = '1'
$env:RAG_ORCHESTRATOR = 'langgraph'
$env:RAG_ENABLE_LLM_PLANNER = '0'
python -m bank_rag.cli ask --knowledge data/demo/generated/knowledge.jsonl --question '星桥演示机构归档需要谁确认？ joint review'
```

Jev 每次证据检查最多发起一次请求，补检后可能再判断一次，SDK 自动重试关闭；默认模型 `jev-1.13.0`，超时 `RAG_JEV_TIMEOUT=45` 秒。缺密钥、缺 SDK、超时或无效响应时，回退原规则 Gate，不额外放行。未新增置信度阈值，保持 Choice 动作为 `answer` 即允许的迁移策略；放行不代表答案正确。不支持的计算仍不会执行。Jev 单独放行时交给生成分支，不把规则缺失的操作数标为已验证。

`evidence_assessment.complete` 保留原规则结果；`allow_to_answer`、`admission_sources` 和 `jev` 记录实际放行及 Jev 输出。只对相同 ID、正文和完整元数据的重复证据去重；不同坐标或同 ID 内容冲突保留。Jev 请求仅包含问题、锁定条件、当前证据和来源定位，不包含 Gold、参考答案或原 Gate 结果。

开启 Jev 会调用收费 TypeSafe API；配置 DeepSeek 密钥后最终生成也可能收费。默认 `scripts/demo.py` 强制关闭 Jev 并移除本进程密钥；只有 `--ask` 才进入问答入口并尊重 Jev 开关。迁移测试使用合成证据和模拟响应，未验证真实服务的语义判断或账户/model 可用性。详见 [Jev 接入说明](docs/jev_integration.md)。

## Evaluation

独立构建的 100 题 Blind Test 经最终复测和逐题裁定，准确率为 **85%**。最终版本复测使用了已经暴露问题的同一题集，不能视为新的完全盲测泛化证明。结果来自私有赛事数据环境；原始材料、完整测试集和 Gold 不公开，demo 不复现 85%。详见 [评测边界](docs/evaluation.md)。

## 文档

- [已知限制](docs/known_limitations.md)
- [公开仓库操作手册](docs/PUBLIC_REPO_MANUAL.md)
- [公开审计](public_repo_audit.md)

## License

本仓库沿用所有者在 GitHub 仓库中选择的 [Unlicense](LICENSE)。第三方依赖遵循各自许可证；赛事资料和私有测试材料不在授权或发布范围内。
