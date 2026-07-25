# MATLAB + Inkscape 正式绘图工作流

本流程面向高完成度、强视觉叙事的论文 Figure 1，同时适用于包含真实数据子面板的
综合图。它把“数据正确性”和“视觉设计”分开管理。

## 1. 不同工具的职责

| 工具 | 负责 | 不负责 |
|---|---|---|
| MATLAB | 数据读取、统计计算、坐标、曲线、柱高、误差区间、热力图 | Figure 1 的自由构图和装饰性图元 |
| Draw.io | 低保真故事板、模块关系、箭头方向 | 最终视觉语言和精细排版 |
| Inkscape | 最终构图、定制 SVG 图元、文字层级、箭头、局部强调和导出 | 修改数据值或凭视觉手调实验几何 |
| Figma | 可选的多人实时协作 | 当前 CEDAR 默认流程 |

核心原则：**MATLAB 是定量图的真值源，Inkscape SVG 是综合 Figure 1 的终稿源。**

## 2. 推荐目录

每张正式图建立独立目录，不覆盖模板或 synthetic demo：

```text
figures/
  figure1/
    README.md
    matlab/
      build_figure1_panels.m
    data/
      README.md
    generated/
      panel_tradeoff.pdf
      panel_mechanism.pdf
    assets/
    figure1.svg
    figure1.pdf
    preview.png
```

- `matlab/`：可重现的数据图代码。
- `data/README.md`：指向冻结结果，不复制或改写证据来源。
- `generated/`：由 MATLAB 生成，不在 Inkscape 中修改其数据几何。
- `assets/`：实际使用且已登记来源的第三方/自建 SVG 图元。
- `figure1.svg`：唯一 Inkscape master。
- `figure1.pdf`：论文使用的矢量交付文件。

## 3. 设计流程

### 阶段 A：先确定一句话

先写出读者看完图后应记住的一句话，再决定面板数量。Figure 1 通常只承载一条主
叙事，其他信息用于解释这条叙事，不应平铺所有模块。

同时收集 3–5 张目标论文图，分别记录：视觉焦点、阅读顺序、主次对比、图文比例和
信息密度。借鉴结构，不复制受版权保护的具体成图。

### 阶段 B：低保真故事板

用纸笔或 Draw.io 只确定：

1. 读者从哪里开始阅读；
2. 哪个区域是视觉焦点；
3. 哪些内容必须是数据，哪些是概念示意；
4. 面板之间的因果、对比或时间关系。

这一阶段不用默认图标和卡片填满画布，也不做最终配色。

### 阶段 C：MATLAB 生成定量面板

所有数据必须来自正式结果或明确批准的新分析。推荐优先导出矢量 PDF：

```matlab
exportgraphics(gcf, "generated/panel_tradeoff.pdf", ...
    "ContentType", "vector", ...
    "BackgroundColor", "none");
```

需要 SVG 且当前 MATLAB 版本支持时，可测试：

```matlab
print(gcf, "generated/panel_tradeoff.svg", "-dsvg");
```

导出后检查字体替换、透明度、图例、裁切和轴线。若 `exportgraphics` 不能满足要求，
再考虑 `export_fig`，不要同时引入多套导出链。

### 阶段 D：在 Inkscape 建立终稿 master

1. 按论文最终宽度建立 SVG 页面；AAAI 双栏目标约为 `6.975 in`，但投稿前必须
   重新核对 Author Kit。
2. 导入 MATLAB PDF/SVG，将每个数据面板作为命名分组管理。
3. 从已登记的素材库导入 SVG，或按统一规范自绘缺失图元。
4. 用图层区分背景、数据面板、概念图元、连接关系、文字和标注。
5. 统一字体、字号、线宽、圆角、箭头头部、配色和留白。
6. 锁定数据几何分组；数据改变时回到 MATLAB 重新生成并整体替换。

所谓“精修”主要包括：重建视觉层级、定制矢量图元、曲线箭头、对齐、局部高亮、
蒙版、留白和文字层级。它不包括手改数据点、置信区间或柱高。

### 阶段 E：最终导出

保持文字可编辑的 master：

```text
figure1.svg
```

Inkscape CLI 可用时导出 PDF：

```powershell
inkscape figure1.svg --export-filename=figure1.pdf
```

不要把转曲后的 PDF 当作唯一源文件。若投稿环境字体兼容存在问题，应保留原始 SVG
和所有字体信息，再针对交付副本处理。

## 4. 质量检查

### 科学与语义

- [ ] 概念示意、synthetic 数据和正式实证结果已清晰区分。
- [ ] 图中数值与冻结结果文件一致。
- [ ] 未把行为指标扩写为未经验证的内部机制结论。
- [ ] 数据面板可由 MATLAB 脚本重新生成。

### 最终尺寸与排版

- [ ] 在论文实际插入宽度查看，而不是只看放大的编辑画布。
- [ ] 最小可见文字不低于 9 pt。
- [ ] 字体家族、数学字体、线宽、圆角和箭头样式统一。
- [ ] 无文字重叠、连线穿字、裁切和异常留白。
- [ ] 阅读顺序明确，第一视觉焦点对应图的主 claim。

### 可访问性与导出

- [ ] 不仅依靠颜色区分类别，同时使用位置、线型、符号或直接标注。
- [ ] 灰度打印仍能理解主要对比。
- [ ] PDF 保持矢量，无低分辨率截图。
- [ ] 字体已嵌入且没有 Type 3 字体。

当前环境已有检查命令：

```powershell
pdfinfo figure1.pdf
pdffonts figure1.pdf
```

`pdffonts` 输出中的 `emb` 应为 `yes`，字体类型不应为 Type 3。

## 5. 插入论文

最终 PDF 在外部生成并裁好画布，在 LaTeX 中只做普通缩放：

```latex
\includegraphics[width=\textwidth]{figures/figure1/figure1.pdf}
```

不要在 AAAI 主文件中重新拼复杂 TikZ，也不要依赖 `trim`/`clip` 修正画布。图注中的
每项结论都应能定位到图内视觉证据或论文正文的正式结果。

