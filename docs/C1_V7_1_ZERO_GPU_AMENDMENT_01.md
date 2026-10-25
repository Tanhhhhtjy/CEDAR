# C1/V7.1 零 GPU 修正案 01：跨平台 JSONL 哈希

状态：运行前修正；尚未运行 Qwen、GPU、capping 或生成任何实验输出。

日期：2026-07-22。

## 1. 触发原因

Linux 实验 worktree 在静态验证阶段报告：

- 实际原始文件 SHA-256 以 `c712254a` 开头；
- 首次冻结预期 SHA-256 以 `8c34262a` 开头；
- 144 行、N/AO/S 各48、来源映射和逐行 JSON 内容均一致。

核查确认，同一 JSONL 在 Git blob/Linux checkout 中使用 LF，在最初 Windows 工作区使用
CRLF。首次冻结错误地把原始文件字节 SHA 当作数据科学身份，因此换行策略变化被误判为数据
变化。硬闸正确停止了实验，没有任何模型输出需要废弃或重跑。

已知原始文件 SHA：

```text
LF:   c712254afe7f3093763cad9a262a5000d94cedb6a435fe91348d842caf19774c
CRLF: 8c34262a76c0d956e92659975c800ed951355d02944cf706f2fb45a800b7f7fa
```

## 2. 修正

科学身份改为：

```text
SHA-256(canonical JSON serialization of the parsed ordered row array)
```

规范化规则为 UTF-8、对象键排序、无非必要空白、保持行顺序。换行符不进入该表示。

冻结内容哈希：

```text
V7 source: a25577a3e6dcd117254efebe44fcd274307872410438461650d2a0f4137e1a15
V7.1 data: f416d576407ff4e237554009b20458977ff4604116852a54986afbe593c9aa91
```

原始文件 SHA 继续写入 manifest，仅作诊断，不作跨平台通过/失败条件。

## 3. 修正范围

修改：

- builder 写入固定 LF，并同时记录内容哈希和原始文件哈希；
- validator 与 runner 用规范化内容哈希执行硬闸；
- preflight 中的文本代码哈希同样规范化换行符，避免 runner 在另一平台重复误报；
- preflight manifest 更新哈希策略和代码哈希；
- 新增 LF/CRLF 内容等价测试。

不修改：

- 144 条 JSON 对象、顺序、ID、arm、prompt、答案或 family；
- V7 来源数据；
- 模型 revision、thinking、greedy、seed、GPU、p25 或 layers 46–53；
- smoke/formal 规模、统计量、bootstrap、解释矩阵或停止条件。

## 4. 审计边界

本修正由静态哈希失败触发，当时没有运行或查看任何模型行为。它不能改变实验方向，也不允许
依据未来输出继续修改数据或 prompt。
