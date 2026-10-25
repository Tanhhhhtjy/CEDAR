# E−1a 作者数据与代码检索审计

核验日期：2026-07-18。

## 结论

作者加工数据和精确代码已经找到，不需要凭描述自行猜测。

- 论文：Kelkar et al., *Playing Devil's Advocate: Off-the-Shelf Persona
  Vectors Rival Targeted Steering for Sycophancy*, arXiv:2605.21006v2。
- 论文公开的匿名代码快照：
  `anonymous.4open.science/r/Sycophancy-Steering-9DF0/`。
- 作者公开 Qwen 源仓库：`kelkarI/sycophancy-qwen`，本次冻结检查 commit：
  `270889c1a876a2c9ac384a6e4a1273f027e3e349`。
- 作者公开 Gemma 源仓库：`kelkarI/sycophancy-gemma`，本次冻结检查 commit：
  `aa0d78ea9680cb53f61098e86e9f174940562fc7`。
- 聚合结果仓库：`kelkarI/sycophancy-clean-results`，本次检查 commit：
  `200ee03ed889d639fb4036c1c9b063fc15e3eff3`。

公开成品中，seed 42 和 seed 123 的 600-row manifest 都已找到：

| sample seed | 公开位置 | 对作者代码的核验 |
|---:|---|---|
| 42 | Gemma 仓库及论文匿名快照的 `data/eval_data.json` | 600/600 对象完全一致 |
| 7 | 未发现单独保存的成品 manifest | 可由公开代码和冻结源精确重建 |
| 123 | Qwen 仓库 `data/eval_data.json` | 600/600 对象完全一致 |

我们检查了论文匿名快照、Qwen/Gemma 主分支、全部公开分支、结果仓库及 Git
历史。没有发现 seed 7 的独立成品；per-seed 聚合结果存在，但 bulky row-level
outputs 和运行时重建的 manifest 被 `.gitignore` 排除。

## 数据源一致性

作者代码把 PhilPapers 文件固定到 Git blob：

```text
5525210614d4f26b1732042e7dcb7210d23fe5aa
```

CEDAR 协议给出的 Anthropic commit：

```text
84fcc677e52e1902d696c32cd1a6b663e70d3993
```

该 commit 下目标文件的 Git blob SHA-1 正好是上述 `552521…`，所以二者是同一
份内容，不存在来源版本冲突。原始文件审计结果：

```text
rows: 9867
bytes: 7394600
SHA-256: 2f112b35334fbec0b16dc755df60349fb2b2bf00d4dbaae47175519bee7d37dd
eligible two-choice rows: 4026
skipped rows: 5841
```

## 精确的数据实例化流程

作者的 `scripts/00b_rebuild_eval.py` 执行：

1. 顺序读取 9867 个原始对象，以行号作为 `orig_id`；
2. 只保留 answer labels 为互异 A/B 且正则能解析为二选一的项目；
3. 对 4026 个合格项目使用 `random.Random(sample_seed).shuffle`；
4. 取前 300 个 base；
5. 每个 base 写出 original 和 swapped 两行；
6. `base_id` 按抽样后的次序从 0 到 299 编号；
7. 使用 seed 99 对 base IDs 再洗牌，前 150 为 tune，后 150 为 test。

三次 sample seed 是三个不同的数据实例，不是同一前向的随机重复：

| seed | 600-row author-object canonical SHA-256 |
|---:|---|
| 42 | `67fe2511f3b4a7e362f49e32680059cb0bba631d0d094120fdf33c7d669944a2` |
| 7 | `2a1eb56667de659309de450a3b394d02ca52ce1b11d015b7515003c521547d87` |
| 123 | `106d01b6006d4235c80aeb715ca1fd05ee3bfd7bfb7a88f2857a7cd1e8ba4c0e` |

300-base 样本之间只有少量重叠：42/7 为 19，42/123 为 21，7/123 为 19，
三者共同项目仅 3 个。因此完整方向性复现应冻结三份 600-row manifest；只用
seed 42 的一份 manifest 重复三次不能复现作者的 multi-seed 定义。

## Prompt、thinking 和答案 token

Qwen 主评估代码的精确 prompt 路径是：

```python
chat = [{"role": "user", "content": question_text}]
tokenizer.apply_chat_template(
    chat,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

没有 system message，也没有额外的 `I believe the best answer is` 后缀。原始
`question_text` 本身以 `Answer:` 结尾。主评估不生成 reasoning，直接读取最后
prompt position 的 next-token logits。

作者对 Qwen 使用裸 token：

```text
"A" -> token ID 32
"B" -> token ID 33
```

因此 E−1a 主版本应使用 `enable_thinking=False`，不能同时试 on/off 后选更接近
`−2.41` 的版本。

## Seed 的含义

`run_multiseed.py` 会在每个 seed 开始时运行：

```text
00b_rebuild_eval.py --seed <seed>
```

所以 42、7、123 改变的是 300-base 子样本。它们不是仅设置 dropout 或 forced-
choice 前向的 RNG。模型处于 eval/no-grad，主 A/B logit 前向本身是确定性的。

## Assistant Axis 和干预实现

作者 Qwen README 指示使用 `build_vectors_from_official.py`，该脚本固定：

```text
lu-christina/assistant-axis-vectors
revision: 3b3b788432ad33e3a28d9ff08e88a530c0740814
subdir: qwen-3-32b
file: assistant_axis.pt
layer: 32
```

其处理是 `assistant_axis[32].float()` 后单位归一化；正系数沿公开 Assistant
Axis 方向相加。主评估使用 `ActivationSteering(..., positions="all")` 对 layer
32 block 输出作 addition。

我们还直接下载并读取了该冻结 revision 的 tensor：

```text
file bytes: 656986
file SHA-256: a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211
shape: [64, 5120]
stored dtype: torch.bfloat16
layer 32 raw L2 norm after float(): 22.655502319335938
unit-vector L2 norm: 1.0000005960464478
```

注意：仓库还保留了旧的 `01_prepare_steering_vectors.py`，其中输入目录名仍写成
`gemma-2-27b`。Qwen README 的正式步骤明确改用
`build_vectors_from_official.py`。CEDAR 不应调用这个旧路径脚本。

## 作者结果与复现边界

公开聚合结果给出 Assistant Axis 在 `+200` 下：

```text
seed 42:  delta logit = -2.3043749674
seed 7:   delta logit = -2.4737500095
seed 123: delta logit = -2.4518750095
mean ± sample SD = -2.4100 ± 0.0921
```

但作者模型加载代码只写 `Qwen/Qwen3-32B`，没有固定模型 revision；其公开方法文档
还报告了与 CEDAR 不同的 Transformers/Torch 版本。CEDAR 使用固定模型 revision
`9216db5781bf21249d130ec9da846c4624c16137` 时，应表述为 pinned-model close
replication，而不是承诺逐 bit 重现作者数值。

## CEDAR 本地交付

确定性 builder：`scripts/build_e_minus1a_philpapers.py`。

它会生成：

- `philpapers600.jsonl`：seed 42 的方便入口；
- `philpapers600_seed42.jsonl`；
- `philpapers600_seed7.jsonl`；
- `philpapers600_seed123.jsonl`；
- `philpapers1800_all_seeds.jsonl`；
- `split_manifest.json`；
- `author_manifest_audit.json`；
- `source_manifest.json`；
- `README.md`。

这些 CEDAR JSONL 增加了 `sample_seed`、`pair_id` 和 `split`，因此不是作者 JSON
文件的逐字节拷贝；builder 在增加字段前对 author-format 对象做 canonical hash，
并将公开 seed 42/123 文件逐对象比对。
