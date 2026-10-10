# 银行业监管制度与统计报表可信 RAG

面向银行业监管制度与统计报表的可信 RAG 问答系统。目标是保留可追溯证据、处理复杂表格，并在条件不足时避免无依据回答。

## Repository Scope

This public repository contains code, system design and demo data only.
Competition-provided data and confidential materials are excluded.

原项目保持冻结。本目录是独立公开副本；不包含比赛数据、题库、Gold、私有索引、答案或 trace。

## 架构与技术

文档解析 → 索引 → BM25 + Embedding → RRF 融合 → Cross-Encoder → 来源关联与文件内检索 → 证据筛选 → 条件判断 / Gate → 答案生成 → 来源引用。

LangGraph 编排流程；Haystack 维护检索组件；Sentence Transformers 提供 Embedding 和重排；DeepSeek 用于生成；确定性规则处理部分表格选择与计算。公开源码保留原 `src/bank_rag` 结构。详见 [架构](docs/architecture.md)。

私有评测环境曾使用 Gate OR Jev 的外部评测封装；本公开入口没有集成该私有封装，不声称普通 CLI 可复现同一最终运行路径。

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

## Evaluation

独立构建的 100 题 Blind Test 经最终复测和逐题裁定，准确率为 **85%**。最终版本复测使用了已经暴露问题的同一题集，不能视为新的完全盲测泛化证明。结果来自私有赛事数据环境；原始材料、完整测试集和 Gold 不公开，demo 不复现 85%。详见 [评测边界](docs/evaluation.md)。

## 文档

- [已知限制](docs/known_limitations.md)
- [公开仓库操作手册](docs/PUBLIC_REPO_MANUAL.md)
- [公开审计](public_repo_audit.md)

## License

尚未选择开源许可证。本仓库可供代码阅读；在所有者选择许可证前，不声明授予开源再分发许可。第三方依赖遵循各自许可证。
