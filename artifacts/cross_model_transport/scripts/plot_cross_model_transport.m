function plot_cross_model_transport()
% Build Figure 2D and a paired family audit from frozen CSV inputs only.

scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);
figureDir = fullfile(rootDir, 'figures');
if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

forest = readtable(fullfile(rootDir, 'figure2d_forest_input.csv'), 'TextType', 'string');
family = readtable(fullfile(rootDir, 'cross_model_family_effects.csv'), 'TextType', 'string');

navy = [31 64 84] / 255;
teal = [14 140 122] / 255;
armColors = [154 163 173; 94 124 153; 14 140 122; 220 84 42] / 255;
arms = ["N" "R" "V" "F"];
models = ["Qwen-FC" "Llama"];
markers = {'o', 'd'};

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 6.9 3.6]);
ax = axes(fig);
hold(ax, 'on');
desiredEstimands = ["overall_delta_m_general" "v_minus_f_continuous_selectivity_change"];
y = 4;
yPositions = [];
yLabels = strings(0);
for estimandIndex = 1:numel(desiredEstimands)
    for modelIndex = 1:numel(models)
        mask = forest.estimand == desiredEstimands(estimandIndex) & forest.model == models(modelIndex);
        row = forest(mask, :);
        if height(row) ~= 1
            error('Expected one forest row for %s / %s.', models(modelIndex), desiredEstimands(estimandIndex));
        end
        color = navy;
        if estimandIndex == 2
            color = teal;
        end
        plot(ax, [row.ci_low row.ci_high], [y y], '-', 'Color', color, 'LineWidth', 1.35);
        scatter(ax, row.estimate, y, 48, markers{modelIndex}, ...
            'MarkerEdgeColor', color, 'MarkerFaceColor', color, 'LineWidth', 1.0);
        yPositions(end + 1) = y; %#ok<AGROW>
        if estimandIndex == 1
            label = models(modelIndex) + "  overall \Delta m_{general}";
        else
            label = models(modelIndex) + "  V-F selectivity";
        end
        yLabels(end + 1) = label; %#ok<AGROW>
        y = y - 1;
    end
end
xline(ax, 0, ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.9);
yticks(ax, fliplr(yPositions));
yticklabels(ax, fliplr(yLabels));
ylim(ax, [0.5 4.5]);
xlabel(ax, 'Within-model p25 - uncapped effect (logits)');
title(ax, 'Matched forced-choice transport');
grid(ax, 'on');
ax.GridAlpha = 0.14;
box(ax, 'off');
set(findall(fig, '-property', 'FontName'), 'FontName', 'Arial');
set(findall(fig, '-property', 'FontSize'), 'FontSize', 9);
print(fig, fullfile(figureDir, 'figure2d_cross_model_transport.svg'), '-dsvg');
exportgraphics(fig, fullfile(figureDir, 'figure2d_cross_model_transport.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(figureDir, 'figure2d_cross_model_transport.png'), 'Resolution', 600);
close(fig);

pairedFig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 6.9 3.7]);
layout = tiledlayout(pairedFig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for modelIndex = 1:numel(models)
    ax = nexttile(layout, modelIndex);
    hold(ax, 'on');
    rows = family(family.model == models(modelIndex), :);
    for armIndex = 1:numel(arms)
        armRows = rows(rows.arm == arms(armIndex), :);
        x0 = armIndex - 0.12;
        x1 = armIndex + 0.12;
        for rowIndex = 1:height(armRows)
            plot(ax, [x0 x1], [armRows.m_general_uncapped(rowIndex) armRows.m_general_p25(rowIndex)], ...
                '-', 'Color', 0.72 * armColors(armIndex, :) + 0.28, 'LineWidth', 0.55);
        end
        scatter(ax, repmat(x0, height(armRows), 1), armRows.m_general_uncapped, 28, markers{modelIndex}, ...
            'MarkerEdgeColor', armColors(armIndex, :), 'MarkerFaceColor', 'none', 'LineWidth', 0.9);
        scatter(ax, repmat(x1, height(armRows), 1), armRows.m_general_p25, 28, markers{modelIndex}, ...
            'MarkerEdgeColor', armColors(armIndex, :), 'MarkerFaceColor', armColors(armIndex, :), 'LineWidth', 0.9);
    end
    xticks(ax, 1:4);
    xticklabels(ax, cellstr(arms));
    ylabel(ax, 'Family mean m_{general} (logits)');
    title(ax, models(modelIndex) + " (within-model paired)");
    grid(ax, 'on');
    ax.GridAlpha = 0.12;
    box(ax, 'off');
end
set(findall(pairedFig, '-property', 'FontName'), 'FontName', 'Arial');
set(findall(pairedFig, '-property', 'FontSize'), 'FontSize', 9);
print(pairedFig, fullfile(figureDir, 'cross_model_paired_family_geometry.svg'), '-dsvg');
exportgraphics(pairedFig, fullfile(figureDir, 'cross_model_paired_family_geometry.pdf'), 'ContentType', 'vector');
exportgraphics(pairedFig, fullfile(figureDir, 'cross_model_paired_family_geometry.png'), 'Resolution', 600);
close(pairedFig);
end
