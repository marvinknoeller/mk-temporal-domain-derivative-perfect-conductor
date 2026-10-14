clear all
close all
clc;
c0 = 299792458;
% time study
S = load('Convergence_time.mat');
tau = S.T / c0 * 1e9 ./ S.M(:);                   % time step in ns
fig = figure;
fig.Position = [1000 818 560 420];
set(gcf, 'Color', 'w')
fsize = 26;
loglog(tau(3:end), S.e_Ed(3:end),'^-','LineWidth',3,'MarkerSize',10, 'MarkerFaceColor','w', 'Color','#3B5BA5')
hold on
loglog(tau(3:end),tau(3:end).^5*9e-6,'k--','LineWidth',3)
lgd = legend({'$\mathrm{err}(\tau)$'},'FontSize',fsize, 'Interpreter','Latex','LineWidth',2, ...
    'Location','SouthEast');
ax = gca;
ax.FontSize = fsize;
ax.GridAlpha = .4;
ax.MinorGridAlpha = 0.2;
ax.GridLineStyle = '-';
ax.LineWidth = 2.0;
ax.TickLength = [0.02, 0.2];
ax.XLabel.String = '$\tau$';
ax.XLabel.Interpreter = 'latex';
ax.XLim = [5e-1,3e1];
ax.YLim = [1e-8,1e0];
grid on
saveas(gcf,strcat('plots/','time_order'),'epsc')

%%
clear
S = load('Convergence_space.mat');
h = S.h_mean(:);                                  % mean edge length in m
X = {S.Es, S.Ed};
p = zeros(numel(h) - 2, 2);
for q = 1:2
    d = arrayfun(@(k) max(abs(X{q}{k}(:) - X{q}{k+1}(:))), 1:numel(h) - 1);
    for k = 1:numel(h) - 2
        f = @(x) (h(k)^x - h(k+1)^x) / (h(k+1)^x - h(k+2)^x) - d(k) / d(k+1);
        p(k, q) = fzero(f, [0.5 6]);
    end
end
fprintf('n = %2d, %2d, %2d:  E'' %.3f\n', [S.n(1:end-2), S.n(2:end-1), S.n(3:end), p(:,2)]');