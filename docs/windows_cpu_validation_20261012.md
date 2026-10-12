# Windows CPU 初始化修复验证

2026-10-12，在隔离副本中复现了 Torch 2.14.1 的 `c10.dll` / WinError 1114 导入失败。安装官方 Torch 2.6.0+cpu 后，Torch 导入恢复；最新 Sentence Transformers 6.1 / Transformers 5.19 又出现无加速器错误，因此使用 `configs/windows-cpu-constraints.txt` 中的兼容组合。

验证结果：

- 35 项合成离线测试通过，包括初始化故障返回安全的 503 响应。
- 修复后的 `pip check` 通过。
- 真实缓存模型加载及合成 hybrid 检索通过：Embedding、RRF、Cross-Encoder、目录关联返回 4 条证据。
- 在用户授权发送相关证据后，使用当前单文件知识库和原问题进行一次 in-process FastAPI `/ask` 真实调用，HTTP 200，无初始化或运行时异常。
- DeepSeek 一次生成完成，无截断、空回复回退或生成错误；数值校验为 warning，需要人工核验。
- Jev 返回 TypeSafeAuthenticationError，当前 Key 未通过认证。本次由规则 Gate 放行，不代表 Jev 服务已验证可用。

完整请求结果和状态摘要只写入本地 Git 忽略的 `results/`，不公开证据、答案或密钥。本文件仅记录技术状态，不包含实际资料内容。原项目和公开主副本未修改。

旧 uvicorn 进程需要由用户停止并重新启动，才能加载新依赖和 API 代码。更新 TypeSafe Key 时只在本地 `.env` 中配置，不向聊天或仓库提交密钥。

官方 CPU 安装来源：https://pytorch.org/get-started/previous-versions/#v260
