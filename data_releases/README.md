# 数据归档与下载

[2026-10-25 研究归档 Release](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) 保存 12 个分包，共约 1.17 GB 压缩数据，涵盖 25,670 个来源文件。仓库保持私有，需要有仓库权限才能下载。

每个分包保存原始文件字节和路径，附 `ARCHIVE_FILE_MANIFEST.csv`。本次已核对打包前的源文件哈希，并解开压缩流逐一复核所有成员；预览版代码和报告同时存入 Git。原始数据和历史协议的内容没有被重写。

| 分包 | 来源文件数 | 下载大小 | 内容 |
|---|---:|---:|---|
| [b1b2-20261006](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 1,310 | 145.79 MiB | B1/B2 的协议、代码、24+192+192 条记录及全部 408 份 NPZ；独立复核及 H/JEV 设计。 |
| [c1-failed-v2-20260928](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 86 | 0.36 MiB | 旧 C1 v2 工程失败及诊断，单独保留。 |
| [c1-signed-20261003](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 4,425 | 41.30 MiB | signed-dose C1 的 3892 条 JSON、哈希、协议、分析和收尾证据；3892 份运行 NPZ 仍仅有远程引用。 |
| [cross-model-additional](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 18 | 3.72 MiB | 跨模型原始记录和 Llama 补充数据。 |
| [historical-manuscript](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 90 | 5.51 MiB | 历史 LaTeX、图表和稿件版本。 |
| [historical-server-20260922](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 15,431 | 134.10 MiB | 9 月恢复的早期服务器快照，包括原始 artifacts、数据、代码、文档和测试。 |
| [research-history](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 48 | 0.48 MiB | 早期思想、路线、评议、累积实验记录、论文草稿和参考文献。 |
| [research-notes](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 21 | 0.11 MiB | 研究重审、路线复盘、微调专题、未来方向文献调研及引用核验。 |
| [rg1p-full-20260728](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 2,217 | 743.35 MiB | RG1-P 完整交付，包括原始内部状态 safetensors、数值记录、审计和图表；与转移 tar 的 2217 个文件完全一致。 |
| [s1p1-20260928](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 117 | 1.32 MiB | S1/P1 本地输入、冻结代码、396 条汇总记录、分析和审计；运行 NPZ 不在此目录中。 |
| [supplementary-local-files](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 599 | 29.57 MiB | 从其余实验副本与监控快照中检出的 599 份独有文件；逐源别名见比较清单。 |
| [supporting-data](https://github.com/Tanhhhhtjy/CEDAR/releases/tag/archive-20261025) | 1,308 | 14.77 MiB | 原有匿名化支持包，原始 manifest 的 1308 个文件核验全部一致。 |

## 原始数组的覆盖范围

- **B1/B2**：408 个运行 NPZ 全部在本次归档，逐条重新核对了 JSON 中记录的 SHA-256。
- **RG1-P**：完整本地交付已保存。`safetensors` 在这里承载实验内部状态，保留在 Release 中。
- **signed-dose C1**：3892 条 JSON 和哈希齐全；对应 3892 份运行 NPZ 在此次本地检索范围内未找到，完成报告记载其保存在服务器。C1 分包中的准备阶段 NPZ 不能替代这些运行数组。
- **S1/P1**：396 条汇总记录齐全。跨批次 SHA-256 比对找到与其中 204 条记录匹配的数组字节，位于 B 批归档；剩余 192 条未在此次本地范围找到。匹配位置见 [available_array_aliases.csv](audits/available_array_aliases.csv)。

未找到的 4,084 条数组引用与预期哈希见 [remote_only_array_inventory.csv](audits/remote_only_array_inventory.csv)。这是本地可得性清单；本次没有连接服务器重新确认这些文件现状。完整重算相应原始数组分析前，需要补齐这些来源。

## 下载和验证

先安装并登录 GitHub CLI；选一个分包即可，不必下载全部数据：

```powershell
gh release download archive-20261025 -R Tanhhhhtjy/CEDAR --pattern "b1b2-20261006.tar.gz" --dir D:/CEDAR/downloads
python tools/verify_archive.py D:/CEDAR/downloads/b1b2-20261006.tar.gz
```

验证程序使用 Python 标准库，先检查整个压缩包，再逐文件检查路径、大小、SHA-256 和内置清单。验证时不执行归档中的代码，也不解压到文件系统。

[ASSETS.json](ASSETS.json) 给出分包大小与哈希；[SHA256SUMS](SHA256SUMS) 可供其他工具校验；[manifests/](manifests/) 给出原始来源和每个文件的哈希；[audits/](audits/) 保存本次比较与验证记录。

解包后按各批协议复现。归档保留原来的绝对路径和运行状态字段，用于历史追溯；旧监控入口与旧预算政策不代表当前运行授权。第三方数据和代码仍遵循原来源许可。
