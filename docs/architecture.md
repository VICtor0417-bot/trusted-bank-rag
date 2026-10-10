# 实际代码结构

| 步骤 | 文件 |
|---|---|
| PDF/DOCX/Excel 解析与 JSONL | src/bank_rag/ingest.py |
| 问题解析、名称归一化 | question_parser.py、canonicalization.py |
| 检索、RRF、CE、来源检索、生成 | pipeline.py |
| 表格目录分块、关联行 | catalog_blocks.py、catalog_row_link.py |
| 文件内词法检索 | source_local_lexical.py |
| 表格选择、期间绑定 | table_selection.py、explicit_row_selection.py、period_binding.py |
| 门控及生成策略 | evidence_gate_policy.py、generation_policy.py |
| LangGraph 状态和路由 | graph.py |
| 评测代码 | evaluate.py、evaluator.py、evaluation_formulas.py、evaluation_locators.py |
| 接口 | cli.py、api.py |

上表除首行外均位于 src/bank_rag。目录候选用于发现来源，原始行/条款才是回答依据。全局表格目录与文件内行证据是两个层级。代码保留原实现，不因公开打包重构。

JSONL 每行含 content 和 meta；meta 保存来源、证据类型以及适用的页码、工作表、行或单元格。各类记录字段不完全相同，详细生成逻辑以 ingest.py 为准。
