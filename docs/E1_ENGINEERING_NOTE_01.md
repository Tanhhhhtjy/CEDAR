# E1 工程记录 01：首次 smoke 写盘失败

日期：2026-07-19。

代码提交 `b0c1ec9` 的首次 E1.0 smoke 在第一条 `no_hook` 生成完成后、写入 checkpoint
metadata 时停止。原因是 `finite` 字段的末项由 `numpy.ndarray.all()` 返回 `numpy.bool_`，
Python JSON encoder 不接受该标量类型。异常发生在正式 record JSON 原子替换之前；没有
形成完整 trajectory record，也没有生成 `SMOKE_AUDIT.json`，更未进入 E1.1/E1.2。

处理规则：

- 不读取该未完成生成来判断科学方向；
- 原不完整输出目录整体改名隔离并保留，不覆盖或伪装为成功运行；
- 将 `finite` 显式转换为 Python `bool`；
- checkpoint JSON 写入失败时清理同一次写入产生的孤立 NPZ，防止伪装成可恢复记录；
- 修复后重新提交代码并生成绑定新 commit 的 `INPUT_MANIFEST.json`；
- 最终 `run_manifest.json` 和中文总结披露一次工程重启；
- smoke 重新通过前仍禁止 E1.1/E1.2。

该修复不改变数据、prompt、seed、生成参数、模型、axis、phase、estimand 或统计闸门。
