clear
close all
clc
files = dir('TaylorTest_ref*.mat');
S = load(files(1).name);
e = S.epsilon(:);

%%
fig = figure();
fig.Position = [1000 818 560 420];
set(gcf, 'Color', 'w')
fsize = 26;
loglog(e, S.Rp(:),'^-','LineWidth',3,'MarkerSize',10, 'MarkerFaceColor','w', 'Color','#3B5BA5')
hold on
loglog(e,e.^2*9e-1,'k--','LineWidth',3)
lgd = legend({'$\Delta(\varepsilon)$'},'FontSize',fsize, 'Interpreter','Latex','LineWidth',2, ...
    'Location','SouthEast');
ax = gca;
ax.FontSize = fsize;
ax.GridAlpha = .4;
ax.MinorGridAlpha = 0.2;
ax.GridLineStyle = '-';
ax.LineWidth = 2.0;
ax.TickLength = [0.02, 0.2];
ax.XLabel.String = '$\varepsilon$';
ax.XLabel.Interpreter = 'latex';

grid on

saveas(gcf,strcat('plots/','taylor_order'),'epsc')
