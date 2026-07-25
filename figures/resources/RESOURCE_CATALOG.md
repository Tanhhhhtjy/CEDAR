# 科研绘图资源台账

最后核验：**2026-07-24**

本台账用于回答“某类图应该从哪里开始”，不是安装清单，也不是对第三方项目的
质量背书。GitHub stars 和最近推送日期来自 2026-07-24 的 GitHub API 快照，只能
辅助判断社区规模和活跃度。

## 推荐级别

| 级别 | 含义 |
|---|---|
| **A** | 当前 CEDAR 绘图工作流首选 |
| **B** | 特定图型或局部任务推荐 |
| **C** | 参考、草稿或备选，不进入默认终稿流程 |
| **Avoid** | 解决许可证、格式或工作流风险前不要直接采用 |

所有第三方 Skill 当前均为 **未安装**。实际复制任何模板、SVG 或图标前，必须在
[ASSET_PROVENANCE.md](ASSET_PROVENANCE.md) 登记具体来源和许可证。

## 一分钟选型

| 需求 | 首选 | 备选 |
|---|---|---|
| 真实曲线、柱状图、消融、热力图 | MATLAB 内置绘图 + `exportgraphics` | gramm、export_fig |
| 置信区间/误差带 | MATLAB 原生实现 | shadedErrorBar |
| Figure 1 构图和视觉语言 | Inkscape + ML Visuals 参考 | Draw.io 故事板 |
| 数学公式 | Inkscape + TexText | MATLAB/LaTeX 先生成矢量 PDF |
| 神经网络结构子图 | NN-SVG 或 PlotNeuralNet | OpenTikZ（先解决许可证） |
| 通用 SVG 图标 | Tabler Icons | Material Design / Carbon Icons |
| 最终交付 | Inkscape SVG master + 矢量 PDF | 不以 Draw.io 为终稿 master |

## A：当前核心工作流

| 资源 | 类型与用途 | 许可证/状态 | 编辑格式 | 使用条件与风险 |
|---|---|---|---|---|
| [MATLAB `exportgraphics`](https://www.mathworks.com/help/matlab/ref/exportgraphics.html) | 官方功能；导出定量子面板 | 商业软件；本机 PATH 未发现 | `.m` → PDF/图片 | 数据几何唯一真值来源；不同 MATLAB 版本支持格式可能不同，优先矢量 PDF |
| [MATLAB `print`](https://www.mathworks.com/help/matlab/ref/print.html) | 官方功能；需要 SVG/EPS 时的补充导出 | 商业软件；本机 PATH 未发现 | `.m` → SVG/PDF/EPS | 需要检查字体替换、裁切和透明度 |
| [Inkscape](https://inkscape.org/) | 自由矢量编辑器；Figure 1 最终组合、排版和导出 | GPL-2.0-or-later；本机 PATH 未发现 | SVG/PDF | `figure1.svg` 作为终稿 master；不得手改数据点和曲线形状 |
| [ML Visuals](https://github.com/dair-ai/ml-visuals) | 100+ ML 视觉素材与构图参考 | MIT；17,342 stars；最近推送 2023-02-13 | Google Slides，可导出后重建为 SVG | 适合作为图元和视觉语言来源，不是可直接套用的完整论文 Figure 1；使用具体素材仍需登记来源 |
| [gramm](https://github.com/piermorel/gramm) | MATLAB 的 ggplot2 风格统计绘图库 | MIT；822 stars；最近推送 2026-06-25 | MATLAB 源码 → 图形对象/PDF | 适合分组、分面和统计图；正式采用前用项目数据做兼容性小测试 |
| [export_fig](https://github.com/altmany/export_fig) | MATLAB 高质量裁切和矢量导出 | BSD-3-Clause；1,346 stars；最近推送 2026-07-24 | MATLAB → PDF/EPS/PNG | 当内置导出不能满足透明度、裁切或渲染要求时使用；输出仍需字体检查 |

## B：特定任务组件

| 资源 | 类型与用途 | 许可证/状态 | 编辑格式 | 使用条件与风险 |
|---|---|---|---|---|
| [shadedErrorBar](https://github.com/raacampbell/shadedErrorBar) | MATLAB 置信带/误差带 | LGPL-3.0；182 stars；最近推送 2024-06-13 | MATLAB 源码 | 检查透明度在 PDF 中是否正确；图例和误差定义必须与统计分析一致 |
| [TexText](https://github.com/textext/textext) | Inkscape 中插入、再次编辑 LaTeX/Typst 公式 | BSD-3-Clause；1,090 stars；最近推送 2026-01-06 | SVG 内公式对象 | 仅在图中确实需要公式时安装；需要本地 TeX 环境，最终检查字体一致性 |
| [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) | Python + TikZ 神经网络结构生成器 | MIT；24,915 stars；最近推送 2023-08-21 | Python/TikZ → PDF | 适合 CNN、U-Net、Encoder–Decoder 等规整子图；自由布局 Figure 1 仍需 Inkscape 重组 |
| [NN-SVG](https://github.com/alexlenail/NN-SVG) / [在线工具](https://alexlenail.me/NN-SVG/) | 参数化生成神经网络 SVG | MIT；5,684 stars；最近推送 2026-06-02 | SVG | 适合快速生成网络骨架；导入后统一字体、线宽和颜色 |
| [Tabler Icons](https://github.com/tabler/tabler-icons) | 统一线性 SVG 图标 | MIT；21,223 stars；最近推送 2026-07-23 | SVG | 当前通用图标首选；保持同一图标家族，不混用多个视觉体系 |
| [Material Design Icons](https://github.com/google/material-design-icons) | 通用 SVG 图标 | Apache-2.0；53,648 stars；最近推送 2026-07-24 | SVG | 图标语言偏产品界面；论文中应克制使用并登记具体文件 |
| [Carbon Icons](https://github.com/carbon-design-system/carbon-icons) | IBM Carbon 图标集 | Apache-2.0；94 stars；最近推送 2024-05-23 | SVG | 作为 Tabler 缺少特定语义图标时的备选，不建议混搭风格 |

## C：参考、草稿与辅助工作流

| 资源 | 类型与用途 | 许可证/状态 | 当前判断 |
|---|---|---|---|
| [SciPilot Figure Skill](https://github.com/Haojae/scipilot-figure-skill) | 第三方 Skill；分析数据、选图、Matplotlib 和视觉检查 | MIT；1,377 stars；最近推送 2026-06-15；**未安装** | 可参考其色盲安全和检查流程；CEDAR 正式数据图当前以 MATLAB 为主 |
| [AI-Research-SKILLs / academic-plotting](https://github.com/Orchestra-Research/AI-Research-SKILLs/tree/main/20-ml-paper-writing/academic-plotting) | 第三方 Skill；通用学术绘图建议 | MIT；仓库 11,056 stars；最近推送 2026-06-16；**未安装** | 只作方法参考，不作为 AAAI-27 格式依据 |
| [Inkscape Figures](https://github.com/gillescastel/inkscape-figures) | Inkscape 图文件管理器 | MIT；642 stars；最近推送 2022-01-28 | 不是模板库；只有图很多且管理成本明显时再考虑 |
| [Draw.io Diagram Builder Skill](https://github.com/Will-hxw/drawio-diagram-builder-skill) | 第三方 Skill；生成可编辑 Draw.io 草图 | MIT；202 stars；最近推送 2026-07-02；**未安装** | 适合故事板和关系图，不解决高完成度 Figure 1 的最终设计问题 |
| [Research Figure Composer Skill](https://github.com/shuang-afk/research-figure-composer-skill) | 第三方 Skill；生成 SVG/Draw.io 科研图草稿 | MIT；2 stars；最近推送 2026-05-28；**未安装** | 项目很新，只适合试验性草稿，不能替代人工视觉检查 |
| [TikZ Editor](https://github.com/DominikPeters/tikz-editor) / [在线编辑器](https://tikz.dev/editor/) | 可视化 TikZ 编辑 | MIT；400 stars；最近推送 2026-07-16 | 规整数学/几何子图的备选；不作为自由排版 Figure 1 主工具 |
| [TikZ.net](https://tikz.net/) | 大量 TikZ 示例 | 示例许可证需逐项核验 | 用于寻找语法和局部结构，不直接复制未确认许可证的代码 |
| [Awesome TikZ](https://github.com/xiaohanyu/awesome-tikz) | TikZ 资源索引 | GitHub API 未识别 SPDX；1,789 stars；最近推送 2026-07-14 | 只作为导航入口；跳转后核验实际项目许可证 |
| [Overleaf TikZ Templates](https://www.overleaf.com/latex/templates/tagged/tikz) | 社区模板集合 | 每个模板单独授权 | 可找特殊构图；必须逐模板核验许可证和宏包兼容性 |

## Avoid / 使用前必须解决的问题

| 资源或做法 | 原因 | 解除条件 |
|---|---|---|
| [OpenTikZ](https://github.com/opentikz/opentikz) | 212 stars、最近推送 2026-07-12，但 GitHub API 未识别许可证；现成模板规模也较小 | 找到明确覆盖目标代码/图标的许可证，并记录具体来源后，才可复制到正式图中 |
| 直接套 Draw.io 默认形状作为 Figure 1 终稿 | 已有演示证明其可编辑，但视觉语言容易变成企业流程图 | 仅作故事板，或在 Inkscape 中彻底重构视觉层级和图元 |
| 在 Inkscape 中拖动实验数据点、修改柱高 | 破坏数据可追溯性，图与分析代码失去一致性 | 回到 MATLAB 重新生成整个数据子面板 |
| 混用 Tabler、Material、Carbon 等多套图标 | 线宽、圆角和视觉语气不一致 | 每张正式图选择一个主图标家族，缺失图标按同一规范自绘 |
| 未登记来源的网络 SVG、截图或模板 | 可能存在版权、署名和复用风险 | 在素材台账记录 URL、版本、许可证、修改和使用位置 |
| 把第三方 Skill 的建议视为 AAAI 格式规则 | Skill 可能过时或基于其他会议模板 | 以 AAAI-27 Author Kit 为唯一合规依据 |

## 维护方法

新增资源时至少填写：直接链接、类型、适用图型、推荐级别、许可证、最近核验日期、
可编辑格式、安装状态和一个明确风险。资源进入正式论文图时，还必须在素材来源台账
登记到具体文件或图元级别。

