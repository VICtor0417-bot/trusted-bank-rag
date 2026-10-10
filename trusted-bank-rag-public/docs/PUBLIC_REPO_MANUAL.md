# 公开仓库操作手册

## 1. 这个目录是什么
从冻结项目中复制的独立 GitHub 公开副本。原项目保留且不受影响。公开版仅含代码、重新编写的说明和自建 demo。

## 2. 什么可以上传
本目录经过审计的 src、配置模板、依赖声明、文档、合成 demo、通用 demo 脚本与测试。不要把其他目录直接复制进来。

## 3. 什么绝对不能上传
赛事数据、官方资料和附件、Gold、私有测试集、原始 PDF/Excel、原文 chunk、私有索引、embedding cache、检索片段和 trace、API key、.env、本地绝对路径及个人信息。不明确的文件标记 MANUAL_REVIEW 并不复制。

## 4. 今后同步代码
从原项目复制前依次检查 diff、数据引用、secret、本地路径、输出日志。只复制审核过的实现；不得带入题目常量或私有 fixture。复查后再提交。

## 5. Git 提交前检查
- [ ] 无 API key 或真实凭据
- [ ] 无 .env
- [ ] 无赛事数据、Gold、私有测试集
- [ ] 无赛事原文 chunk、私有索引或 trace
- [ ] 无本地绝对路径
- [ ] README 路径有效
- [ ] demo 可运行
- [ ] 新增文件已复审
- [ ] 保留 LICENSE，检查新增第三方代码的许可证

## 6. 运行 demo
```sh
python -m pip install -e ".[dev]"
python scripts/demo.py
python -m pytest tests
```
默认不调用 API；离线预览不等于完整生成验证。可选 `python scripts/demo.py --ask` 使用自行配置的 DeepSeek 凭据。

## 7. 更新依赖
本项目依赖声明位于 pyproject.toml；不导出包含本地实验环境的整个 pip freeze。更新后在隔离环境执行安装、demo 与测试，记录版本。模型权重不提交。

## 8. Git 提交
在本公开目录单独初始化仓库，先检查待提交文件：
```sh
git init
git status
git add .
git diff --cached --stat
git commit -m "Prepare public code and synthetic demo"
# 配置自己的远端并再次复查后，手动执行：
git push
```
本公开目录已建立独立 Git 仓库。后续提交前继续执行上述检查，不要在原项目的父目录直接提交全部文件。
