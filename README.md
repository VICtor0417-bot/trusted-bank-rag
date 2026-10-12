# 银行业监管制度与统计报表可信 RAG

面向银行监管制度和统计报表的问答系统：解析文件、检索证据、检查条件与证据完整性，再生成带来源引用的回答。当前版本 **0.2.0**。

公开仓库仅包含代码、配置模板、文档和虚构演示资料。原始业务资料、API Key、知识库、模型缓存、答案及 trace 不公开。目前通过本地文件夹录入资料，没有网页上传文件接口。

## 系统流程

文件解析 → JSONL 知识库与文件梗概 → BM25 / Embedding → RRF 融合 → Cross-Encoder 重排 → 来源关联及文件内检索 → 规则 Gate / 可选 Jev → 确定性计算或 DeepSeek 生成 → 答案校验与引用。

LangGraph 编排问答流程，Haystack 连接检索组件。混合检索使用 `BAAI/bge-small-zh-v1.5` 和 `BAAI/bge-reranker-base`。Jev 采用“规则 Gate 通过 **或** Jev 判断 answer 即放行”；不支持的计算仍被拒绝。放行与引用存在均不等于答案正确。详见 [架构](docs/architecture.md) 和 [Jev 接入](docs/jev_integration.md)。

## 1. 下载项目并安装环境

以下面向 **Windows PowerShell + Python 3.12 + CPU**。先安装 Python 和 Git，再执行：

```powershell
git clone https://github.com/VICtor0417-bot/trusted-bank-rag.git
Set-Location .\trusted-bank-rag
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-compile torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install --no-compile -e ".[dev,jev]" -c .\configs\windows-cpu-constraints.txt
.\.venv\Scripts\python.exe -m pip check
```

已下载项目时，进入包含本 README 的目录即可。以上命令不需要激活虚拟环境；每个项目应有自己的 `.venv`，不要复制其他项目的虚拟环境。

CPU 约束固定了本次验证可工作的 Torch / Sentence Transformers / Transformers 组合。全新机器的完整安装与首次模型下载尚未验证。GPU 用户请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v260) 选择适配版本，不使用 CPU 约束。

Linux/macOS 可激活虚拟环境后执行 `python -m pip install -e ".[dev,jev]"`，后续将 Python 路径替换为 `python`，自行选择兼容的模型依赖。

## 2. 放入自己的文件

克隆后的固定目录：

```text
项目根目录/
├─ raw/                           放原始文件，支持子目录
├─ data/
│  ├─ generated/                  生成自己的知识库和梗概
│  └─ demo/                       公开的虚构演示资料
├─ results/                       保存问答结果
└─ .env                           自己创建的本地配置
```

把文件复制到项目根目录的 `raw/`，不要放进另一个同名目录。支持 PDF、DOC、DOCX、XLS、XLSX、XLSM。扫描图片型 PDF 需先 OCR，本项目不执行 OCR。Excel 读取缓存值，不重新计算公式。来源按文件标题分组，尽量避免不同文件使用相同的不含扩展名文件名。

仓库随附 `raw/001_2099年银行业总资产、总负债（月度）_虚构测试.xlsx`，数值由用户编造，标题与页脚已注明虚构。仅 1、2 月有数据，其余月份为空。按下面的入库、梗概和启动步骤即可测试；问题可用“2099年1月银行业金融机构总资产是多少？”，原表对应值为 **1234 亿元**。该值用于核对结果，不保证系统每次都能正确回答。单独解析此文件已验证生成 96 条记录、无解析错误。

导入自己的真实资料前，可把这份虚构文件移到 `raw/` 之外，避免一起入库；不要只移到 `raw/` 的子目录，因为入库会递归扫描。

旧 `.doc` 需要 antiword 或 LibreOffice。安装 LibreOffice 后，可设置转换程序路径：

```powershell
$env:SOFFICE_PATH = 'C:\Program Files\LibreOffice\program\soffice.exe'
```

检查输入数量：

```powershell
$inputFiles = @(Get-ChildItem -LiteralPath .\raw -File -Recurse |
  Where-Object { $_.Extension.ToLowerInvariant() -in '.pdf','.doc','.docx','.xls','.xlsx','.xlsm' })
$inputFiles.Count
```

数量为 0 时先检查目录和扩展名。建议先放一份文件跑通，再扩充资料。

## 3. 解析入库和生成文件梗概

```powershell
.\.venv\Scripts\python.exe -m bank_rag.cli index `
  --source .\raw `
  --output .\data\generated\knowledge.jsonl `
  --strategy baseline

.\.venv\Scripts\python.exe -m bank_rag.cli summarize `
  --knowledge .\data\generated\knowledge.jsonl `
  --output .\data\generated\file_summaries.jsonl
```

这两条命令不调用付费 API。`indexed_records` 是证据记录数，**不是文件数**；一张表可能生成许多记录。入库阶段不生成 Embedding。`baseline` 保留正文、表格行、单元格和证据组；`hierarchical` 是可选入库策略，并非完整问答的必需步骤。

`summarize` 默认按规则生成梗概。梗概应与知识库放在同一目录，命名为 `file_summaries.jsonl`。可选 `--llm` 会调用 DeepSeek，需先配置密钥。

重新 `index` 会重写指定知识库。解析错误可能被写成错误记录，非零记录数不代表所有文件都解析成功，应检查来源覆盖和解析错误。先查看记录数：

```powershell
(Get-Content -Encoding utf8 .\data\generated\knowledge.jsonl | Measure-Object -Line).Lines
```

## 4. 在本地 .env 配置 Key 和完整功能

首次创建配置：

```powershell
Copy-Item .\.env.example .\.env
notepad .\.env
```

已有 `.env` 时不要重新复制覆盖。真实密钥只填进本地 `.env`，不要填进 `.env.example`，不要提交 GitHub 或贴到聊天。以下密钥占位文字需替换：

```dotenv
DEEPSEEK_API_KEY=填写自己的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
TYPESAFE_API_KEY=填写自己的TypeSafe密钥
RAG_JEV_MODEL=jev-1.13.0
RAG_JEV_TIMEOUT=45
KNOWLEDGE_PATH=data/generated/knowledge.jsonl
RAG_RETRIEVAL_MODE=hybrid
RAG_ORCHESTRATOR=langgraph
RAG_ENABLE_LLM_PLANNER=1
RAG_ENABLE_JEV=1
RAG_REUSE_INITIAL_GLOBAL=0
HAYSTACK_TELEMETRY_ENABLED=false
```

模型名称需与供应商和账号支持的模型一致。`hybrid` 启用 Embedding、融合及重排；planner 开关启用问题规划；Jev 开关启用证据判断。排查时可把 planner 设为 `0`，减少外部调用，规则解析仍工作。不用 Jev 时设 `RAG_ENABLE_JEV=0`。默认 `.env.example` 使用 BM25 并关闭 planner/Jev，需按上例修改以启用完整模式。

程序不会自行读取 `.env`；本 README 使用 `python-dotenv` CLI 加载它，Key 无需写入命令行。`dotenv run` 默认以 `.env` 覆盖 shell 的同名变量；希望保留 shell 配置时使用 `run --no-override -- ...`。修改 `.env` 后需重启服务。

生成、LLM 规划、补检改写、LLM 梗概和 Jev 会把相关问题及必要内容发到供应商，并可能收费。关闭 planner/Jev 不会关闭已配置的 DeepSeek 生成。Jev 出错会回退规则 Gate，不代表 Jev 已可用。

## 5. 启动 API 和提交问题

在项目根目录执行，保持终端开启：

```powershell
.\.venv\Scripts\python.exe -m dotenv -f .env run -- `
  .\.venv\Scripts\python.exe -m uvicorn bank_rag.api:app `
  --host 127.0.0.1 --port 8000
```

看到 `Uvicorn running on http://127.0.0.1:8000` 后，打开 [Swagger 操作界面](http://127.0.0.1:8000/docs)。

1. 展开 `POST /parse-question`，点 **Try it out**，输入 `{"question":"你的问题"}`，点 **Execute**，查看规则解析。此接口不检索、不调用 Jev/DeepSeek。
2. 展开 `POST /retrieve`，用同样格式提交问题，查看证据和引用。此接口不执行 Jev 或生成。
3. 展开 `POST /ask`，提交问题，查看答案、引用、证据判断和校验状态。此入口执行完整问答；某些目录查询或确定性取数有自己的捷径。

`/health` 只检查服务可响应，不证明知识库、模型或外部服务准备好。第一次 `/retrieve` 或 `/ask` 才加载知识库、模型并建立内存索引；首次下载会更久。后续复用同一进程实例。本次单文件 CPU 完整测试约 69 秒，包含初始化与服务调用，不是其他机器的固定耗时。

不要连续重复提交。停止服务按终端 **Ctrl+C**。更新资料后重新入库和生成梗概，修改配置/Key 后重启服务。CLI 每次启动新进程，连续问答推荐 API。

### 保存 API 结果

服务运行时，在另一个 PowerShell 终端进入项目目录：

```powershell
$question = Read-Host '输入问题'
$body = @{ question = $question } | ConvertTo-Json -Compress
$response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/ask' `
  -Method Post -ContentType 'application/json; charset=utf-8' `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
$response | ConvertTo-Json -Depth 100 |
  Out-File -Encoding utf8 .\results\answer.json
$response.answer
```

结果保存在 `results/answer.json`，后续同名运行会覆盖，需要保留时更换文件名。

### CLI 问答

```powershell
$question = Read-Host '输入问题'
.\.venv\Scripts\python.exe -m dotenv -f .env run -- `
  .\.venv\Scripts\python.exe -m bank_rag.cli ask `
  --knowledge .\data\generated\knowledge.jsonl --question "$question" |
  Out-File -Encoding utf8 .\results\answer.json
```

将 `ask` 替换为 `retrieve` 可只检索。LLM 梗概使用同样的 dotenv 前缀，再运行第 3 节 `summarize` 并加 `--llm`。

## 6. 判断结果是否可用

- 核对 `answer`、`citations` 和原始文件的时间、指标、单位及坐标。
- `evidence_assessment.complete` 是规则 Gate 结果；`allow_to_answer`、`admission_sources` 是实际放行，`jev` 是 Jev 状态。
- `verified: false` 或数值校验 warning 表示不能当作已验证答案。`verified: true` 也需要结合计算过程和引用理解。
- `DELIVERY_ERROR`、`runtime_error`、`generation_incomplete` 表示未交付完整可靠答案。HTTP 200 本身不代表成功。
- 异常响应的 `insufficient_evidence: false` 是默认字段，不能据此认定证据充分。

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| 入库 0 条 | 核对当前项目 `raw/`、扩展名和输入计数。 |
| 找不到知识库 | 先入库，并在 `.env` 指定 `KNOWLEDGE_PATH`；未配置时 API 仍指向 demo 默认路径。 |
| `c10.dll` / `WinError 1114` | 停止服务，按第 1 节安装 CPU 依赖与兼容约束，`pip check` 后重启。 |
| HTTP 503 / `rag_initialization_failed` | 查看终端，检查依赖、模型缓存/下载和知识库。 |
| 首次问答慢 | 首次建立索引和加载模型，不要重复提交。 |
| `APITimeoutError` | 外部模型超时；上一请求结束后可重试一次，持续失败需查看终端 Traceback 和供应商状态。代码没有自定义 DeepSeek 超时环境变量，随意在 `.env` 增加变量不会生效。 |
| completion 被截断 | 输出达到长度上限，检查最终 `generation_incomplete`，不能将截断内容当完整答案。 |
| `TypeSafeAuthenticationError` | 检查本地 TypeSafe Key，更新后重启；规则回退成功不代表 Jev 正常。 |
| jieba / PromptBuilder 警告 | 单独出现不表示失败，以最终响应和后续异常为准。 |
| 端口占用 | 在原服务终端 Ctrl+C，或更换端口并修改浏览器地址。 |

Hugging Face 模型缓存可通过 `HF_HOME` 指定。模型已经完整缓存时，可设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 禁止模型联网检查；首次下载不要启用离线模式。模型权重不提交。

## 离线 demo 与测试

```powershell
.\.venv\Scripts\python.exe scripts/demo.py
.\.venv\Scripts\python.exe -m pytest tests
```

默认 demo 使用自建 DOCX/XLSX、真实 parser、BM25 和引用，关闭 Jev，不加载 Embedding/重排，不调用付费 API。答案是抽取预览，LangGraph 只做构建检查，不能代替完整问答。部分英文标签用于 BM25 匹配，不证明中文检索质量。`--ask` 才进入真实问答，需要通过 dotenv 加载凭据并承担调用成本。

本版本已有 35 项合成离线测试、依赖检查、缓存真实模型检索及一次授权的 DeepSeek 问答验证。该次 Jev 认证失败、数值校验 warning，不证明 Jev 或答案质量已验证。详见 [CPU 验证记录](docs/windows_cpu_validation_20261012.md)。

## 发布与资料保护

`.env`、本地原始资料、`data/generated/`、`results/`、模型缓存、Python 缓存已忽略。资料目录只提交 `.gitkeep`，另为上文明确命名的虚构测试 Excel 设置一条公开例外；其他 `raw/` 文件仍被忽略。不要把真实资料覆盖写入这个已跟踪的测试文件名，否则 Git 会把它视作可提交修改。公开 demo 的 Markdown/CSV 和脚本保留，自动生成产物不再跟踪。忽略规则不会自动移除历史已跟踪文件；新提交删除也不会删除 Git 历史。

后续提交经过检查的代码、文档和模板：

```powershell
git status --short
# 用 git add 明确列出审核过的文件，不强制加入忽略文件。
git diff --cached --stat
git diff --cached
git commit -m "Describe the reviewed changes"
git pull --rebase origin main
git push origin main
```

出现合并冲突先解决，不使用 force push 绕过。

## 评测与文档

历史私有题集最终复测准确率 85%，题集已经暴露过问题，不是新的完全盲测泛化证明。资料与 Gold 不公开，demo 不复现此成绩。用户报告私有比较中 Gate OR Jev 优于任一单独方案；本公开迁移未提供可独立复现的相同比较，不能据此宣称此版本已验证同样提升。

- [架构](docs/architecture.md)
- [Jev 接入](docs/jev_integration.md)
- [评测边界](docs/evaluation.md)
- [已知限制](docs/known_limitations.md)
- [操作手册](docs/PUBLIC_REPO_MANUAL.md)
- [0.2.0 更新说明](docs/releases/0.2.0.md)

## License

沿用所有者选择的 [Unlicense](LICENSE)。第三方依赖遵循各自许可证，私有资料不属于公开发布或授权范围。
