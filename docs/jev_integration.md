# 公开版 Jev 接入

本次只迁入原项目通用的证据判断 rubric 和 OR 放行策略。未迁入私有评测封装、题库、Gold、答案、索引、trace 或凭据；原项目保持冻结。公开版 `src/bank_rag/jev_gate.py` 不依赖原项目路径。

## 入口与执行顺序

普通 CLI `ask`、FastAPI `/ask` 在 `RAG_ORCHESTRATOR=langgraph` 时进入问答图。检索和目录升级后，先保留来源地精确去重，执行规则 `assess_evidence`，再用相同最终证据调用 Jev。`complete` 仍表示规则结果；实际路由使用 `allow_to_answer = rule_allow OR (successful Jev action == answer)`。两者都未放行时，保留原来的缺失项补检/拒答路径，补检后可能再调用一次 Jev。

普通 `retrieve`、legacy/haystack 回退和最早文件目录捷径不执行 Jev。默认离线 demo 显式关闭 Jev。禁用 Jev 不导入 SDK、不创建网络客户端。

## 配置

| 配置 | 默认 | 说明 |
|---|---|---|
| `RAG_ENABLE_JEV` | `0` | 显式开启后才调用 |
| `TYPESAFE_API_KEY` | 无 | 在 shell 或部署环境安全配置，不自动读取 `.env` |
| `RAG_JEV_MODEL` | `jev-1.13.0` | 保留迁移模型；实际可用性由 TypeSafe 决定 |
| `RAG_JEV_TIMEOUT` | `45` | SDK HTTP 操作超时秒数；不是整条问答链的超时 |

使用 `python -m pip install -e ".[dev,jev]"` 安装可选 SDK。自动重试关闭，一轮证据检查只请求一次。模型不可用、网络失败、SDK 缺失或响应不合法时返回仅含错误类型的状态，原规则结果仍可使用，不输出异常消息或凭据。

## 判断输入和输出

输入只有 `user_question`、`resolved_conditions`、证据正文和来源位置。一次请求同时询问 `action`、`evidence_sufficient`、`missing_user_conditions`。四个动作：`answer`、`needs_evidence`、`needs_clarification`、`unsupported`。迁移保持原策略：只用成功响应的 `action == answer` 增加放行，没有新增概率阈值；其余概率和用量保留用于检查。

Jev 判断当前整组证据能否支持所有必要事实，不执行计算，不重新解释锁定条件，不把主题相关或目录卡当成事实。它仍是模型判断，可能出错。规则通过但 Jev 提出 `needs_clarification` 时，保留 OR 放行并把提醒传给最终生成。无 DeepSeek 密钥时只能返回证据，不生成自然语言事实答案。

`evidence_assessment` 同时保留规则缺失项、`rule_allow`、`jev_allow`、`allow_to_answer`、`admission_sources`、`jev` 和去重记录；决策轨迹按实际放行显示，不把 Jev 放行改写成规则槽已填满。不支持的操作仍被图拒绝。Jev 单独放行不进入确定性计算器，而进入最终生成。

## 验证边界

2026-10-12 本机验证：29 项测试通过；默认 demo 解析 21 条合成记录、4/4 查询返回证据；实际 CLI 在 Jev 开启但缺密钥时成功回退规则 Gate，返回证据和 6 条引用。真实 TypeSafe SDK 通过本地 MockTransport 编码和解析检查，FastAPI `/ask` 通过模拟 Jev/DeepSeek 响应检查。付费 API 调用为 0。版本与范围记录在 `jev_validation_20261012.json`。

`tests/test_jev_gate.py` 使用自建合成证据验证 OR 分支、失败回退、输入隔离、来源保留、补检限制、口径提醒和真实 BM25 问答入口。测试不读取实际密钥，模拟的 Jev 输出只验证程序行为，不证明模型语义质量。默认 demo 检查基础解析、检索和引用；完整 hybrid、真实 TypeSafe/DeepSeek 调用和私有评测结果需单独验证。

为让离线 BM25 检查不依赖加载 ML 组件，Sentence Transformers 导入移到模型 `warm_up` 和多向量目录分支；hybrid 使用的模型和接口保持原配置。

本次公开版 `.venv` 安装了离线 BM25/Jev 测试所需依赖及 editable 包，没有安装 Sentence Transformers 或模型权重；不是按完整依赖声明全新安装后的 hybrid 验证。

SDK 接口依据官方文档： https://docs.typesafe.ai/sdk/python 和 https://docs.typesafe.ai/sdk/python/api/clients/sync 。
