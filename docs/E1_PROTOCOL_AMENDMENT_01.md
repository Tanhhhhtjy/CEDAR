# E1 协议修正案 01：prefill challenge 边界

状态：冻结；本修正案在任何 E1 GPU 输出产生之前提交。

日期：2026-07-19。

## 发现

原协议要求分别渲染 `messages[:-1]` 与完整 `messages`，并要求前者 token IDs 是后者前缀。
在固定 Qwen3 chat template 中，这个假设不成立：当消息列表恰好以历史 assistant 消息结束
时，模板会把该 assistant 内容渲染成带空 `<think>...</think>` 包装的最终 assistant 回合；
当后面仍有 follow-up user 消息时，同一历史 assistant 内容不会得到该包装。因此两次独立
渲染在最后一个历史 assistant 回合内已经分叉。

该问题由零 GPU tokenizer 预检发现；尚未运行 E1 smoke、bridge 或 sampling，也没有观察
任何 E1 科学结果。

## 冻结修正

正式 prompt 仍只渲染一次，且与旧 v6 完全相同：

```python
tokenizer.apply_chat_template(
    row["messages"],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
```

`prefill_challenge` 起点改为：

1. 在上述完整 prompt 中从末尾唯一定位最后一个 follow-up user 的原始 `content`；
2. 向前定位与它相邻的最后一个固定 Qwen user header `<|im_start|>user\n`；
3. header 之前为 `prefill_original`；从该 header 开始到 prompt 末尾为
   `prefill_challenge`；
4. 重新拼接两段必须逐字符等于完整 prompt；两段 token IDs 拼接必须逐 token 等于完整
   prompt IDs；
5. follow-up content 找不到、出现歧义、header 不相邻或 token 拼接不等时立即停止。

该修正只改变 token phase 标签的工程定位方式，不改变 prompt、token IDs、模型输入、题目、
seed、生成参数、estimand 或统计闸门。禁止使用模型输出反推边界。
