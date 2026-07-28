function plot_e2a_family_geometry()
% Plot only archived family-level E2A CSV values; no random coordinates.

scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);
figureDir = fullfile(rootDir, 'figures');
if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

family = readtable(fullfile(rootDir, 'e2a_family_geometry.csv'), 'TextType', 'string');
arm = readtable(fullfile(rootDir, 'e2a_family_arm_geometry.csv'), 'TextType', 'string');

gray = [0.48 0.52 0.56];
lightGray = [0.78 0.80 0.82];
teal = [14 140 122] / 255;
orangeRed = [220 84 42] / 255;

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 6.9 5.3]);
layout = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(layout, 1);
hold(ax1, 'on');
for index = 1:height(family)
    plot(ax1, [1 2], [family.m_general_uncapped(index), family.m_general_p25(index)], ...
        '-', 'Color', lightGray, 'LineWidth', 0.65);
end
scatter(ax1, ones(height(family), 1), family.m_general_uncapped, 34, ...
    'o', 'MarkerEdgeColor', gray, 'MarkerFaceColor', 'none', 'LineWidth', 1.0);
scatter(ax1, 2 * ones(height(family), 1), family.m_general_p25, 34, ...
    'o', 'MarkerEdgeColor', teal, 'MarkerFaceColor', teal, 'LineWidth', 1.0);
xlim(ax1, [0.65 2.35]);
xticks(ax1, [1 2]);
xticklabels(ax1, {'uncapped', 'p25'});
ylabel(ax1, 'Family mean m_{general} (logits)');
title(ax1, 'E2A paired family geometry');
grid(ax1, 'on');
ax1.GridAlpha = 0.14;
box(ax1, 'off');

ax2 = nexttile(layout, 2);
hold(ax2, 'on');
families = unique(arm.family_id, 'stable');
vDelta = nan(numel(families), 1);
fDelta = nan(numel(families), 1);
for index = 1:numel(families)
    vMask = arm.family_id == families(index) & arm.arm == "V";
    fMask = arm.family_id == families(index) & arm.arm == "F";
    vDelta(index) = mean(arm.delta_m_target(vMask), 'omitnan');
    fDelta(index) = mean(arm.delta_m_target(fMask), 'omitnan');
    plot(ax2, [1 2], [vDelta(index) fDelta(index)], '-', ...
        'Color', lightGray, 'LineWidth', 0.65);
end
yline(ax2, 0, ':', 'Color', [0.25 0.25 0.25], 'LineWidth', 0.8);
scatter(ax2, ones(numel(families), 1), vDelta, 38, 'o', ...
    'MarkerEdgeColor', teal, 'MarkerFaceColor', teal, 'LineWidth', 1.0);
scatter(ax2, 2 * ones(numel(families), 1), fDelta, 42, 'd', ...
    'MarkerEdgeColor', orangeRed, 'MarkerFaceColor', orangeRed, 'LineWidth', 1.0);
xlim(ax2, [0.65 2.35]);
xticks(ax2, [1 2]);
xticklabels(ax2, {'V', 'F'});
ylabel(ax2, 'Delta m_{target} (logits)');
title(ax2, 'Continuous V/F family effects');
grid(ax2, 'on');
ax2.GridAlpha = 0.14;
box(ax2, 'off');

set(findall(fig, '-property', 'FontName'), 'FontName', 'Arial');
set(findall(fig, '-property', 'FontSize'), 'FontSize', 9);

print(fig, fullfile(figureDir, 'e2a_family_geometry.svg'), '-dsvg');
exportgraphics(fig, fullfile(figureDir, 'e2a_family_geometry.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(figureDir, 'e2a_family_geometry.png'), 'Resolution', 600);
close(fig);
end
