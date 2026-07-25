# CEDAR figure workspace

本目录保存 CEDAR 的绘图源文件、生成产物、资源索引和质量检查说明。

> `editable-topconf-demo/` 中的三张图只用于测试视觉设计与可编辑性，均为
> **synthetic demo**，不代表 CEDAR 方法、实验结果或论文结论。

## 从哪里开始

| 任务 | 首选入口 |
|---|---|
| 查找模板、库、插件或图标 | [完整资源台账](resources/RESOURCE_CATALOG.md) |
| 制作正式 Figure 1 | [MATLAB + Inkscape 工作流](resources/MATLAB_INKSCAPE_WORKFLOW.md) |
| 引入外部 SVG、模板或图标 | [素材来源登记](resources/ASSET_PROVENANCE.md) |
| 查看已有可编辑演示 | [Draw.io 演示说明](editable-topconf-demo/README.md) |

## 工具分工

- **MATLAB**：所有真实数据点、曲线、柱高、热力图和误差区间的唯一真值来源。
- **Inkscape**：正式 Figure 1 的最终矢量排版和 `figure1.svg` 主文件。
- **Draw.io**：可选的低保真故事板和关系草图，不承担最终视觉设计。
- **Figma**：仅在确实需要多人实时协作时考虑，当前工作流不依赖它。

```text
论文 claim → 故事板 → MATLAB 数据面板 → Inkscape 组合精修 → PDF → AAAI 检查
```

## 当前状态（2026-07-24）

- 资源已登记，但没有因本索引而安装任何第三方 Skill、插件或绘图库。
- 当前 Windows `PATH` 未发现 `matlab`、`inkscape` 或 `drawio` 命令；这不等于
  软件一定未安装，正式使用前应确认实际安装路径。
- `pdffonts` 和 `pdfinfo` 可从 TeX Live 调用，用于检查最终 PDF。
- AAAI 格式要求具有时效性；正式投稿前以
  [AAAI-27 Author Kit](https://aaai.org/authorkit27/) 为唯一格式依据。

## 科学边界

概念图、工作流图和 synthetic demo 必须明确标注。任何实证面板只能读取冻结的
正式结果或经批准的新分析，不得为了视觉效果手工调整数据几何。

