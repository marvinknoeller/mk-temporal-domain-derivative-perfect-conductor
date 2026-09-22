clear all
close all
clc

for ii = 1 : 3
    if ii == 1
        grid_folder = 'Example1Grids/grids0';
        exact_file  = 'Exact Object.msh';
        iter = 13;
    elseif ii == 2
        grid_folder = 'Example1Grids/grids15';
        exact_file  = 'Exact Object.msh';
        iter = 12;
    else
        grid_folder = 'Example1Grids/grids30';
        exact_file  = 'Exact Object.msh';
        iter = 13;
    end
    eScale      = 0.3;       % colour limits for the error, in mesh units
    nIso        = 9;         % number of contour lines (an odd number includes d = 0)
    c           = [0 0 0];   % star centre of the objects
    angTol      = 25;        % crease angle in degrees that counts as a sharp edge
    %% ---------- load ----------
    [nodE, elE] = read_msh_file(fullfile(grid_folder, exact_file));
    [nodR, elR] = read_msh_file(fullfile(grid_folder, sprintf('iteration%d.msh', iter)));
    elE = elE(:,1:3);
    elR = elR(:,1:3);
    %% ---------- error and sharp edges ----------
    d     = signed_distance(nodR, nodE, elE, c);    % negative inside the exact object
    %% ---------- figure ----------
    figure('Color', 'w', 'Position', [80 80 1500 900]);
    trisurf(elR, nodR(:,1), nodR(:,2), nodR(:,3), 'FaceVertexCData', d, ...
    'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceLighting', 'none');
    axis equal vis3d
    axis off
    colormap(bwr(1e3));
    clim([-eScale eScale]);
    cb = colorbar('southoutside');
    cb.FontSize = 34;
    cb.Ticks = [-eScale 0 eScale];
    cb.Position = [0.568666666666666,0.14,0.162000000000001,0.05];
    cb.LineWidth = 2.5;
    exportgraphics(gcf,fullfile('plots',strcat('rec1',num2str(ii),'.png')),'Resolution',300)
    close all
end
%% ---------- local functions ----------
function d = signed_distance(P, pos, tri, c)
% signed distance from the points P to the closed surface (pos, tri), < 0 inside
% a triangle consists of vertices A, B and C
A   = pos(tri(:,1),:);
AB  = pos(tri(:,2),:) - A;
AC  = pos(tri(:,3),:) - A;
n   = outward_normals(pos, tri, c); % all normals for all triangles
d00 = sum(AC.^2, 2).'; % squared distance AC
d01 = sum(AC.*AB, 2).';
d11 = sum(AB.^2, 2).'; % squared distance AB
den = d00.*d11 - d01.^2;
d   = zeros(size(P,1), 1);
% do not run all points at once, instead, get the distance in blocks
kb  = max(1, floor(3e5 / size(tri,1)));             % block size
for i0 = 1:kb:size(P,1)
    id  = i0:min(i0+kb-1, size(P,1));
    Q   = P(id,:);
    wx  = Q(:,1) - A(:,1).';
    wy  = Q(:,2) - A(:,2).';
    wz  = Q(:,3) - A(:,3).';
    dp  = wx.*n(:,1).'  + wy.*n(:,2).'  + wz.*n(:,3).';     % signed distance from P_i to the plane, in which T_j is lying
    % Now we need to check the foots (Lotpunkte) from the points to the planes and then, which foots lie in the triangles 
    d20 = wx.*AC(:,1).' + wy.*AC(:,2).' + wz.*AC(:,3).';
    d21 = wx.*AB(:,1).' + wy.*AB(:,2).' + wz.*AB(:,3).';
    u   = (d11.*d20 - d01.*d21) ./ den;
    v   = (d00.*d21 - d01.*d20) ./ den;
    dt  = min(min(segdist(Q, A, AB), segdist(Q, A, AC)), segdist(Q, A + AB, AC - AB));
    in  = u >= 0 & v >= 0 & u + v <= 1;                 % foot point inside the triangle
    dt(in) = abs(dp(in)); % get the distances of all points inside
    dmin = min(dt, [], 2); % get the smallest distance
    near = dt <= dmin*(1 + 1e-9) + 1e-12;               % near is true for triangles that attain the minimum
    d(id) = sign(sum(dp .* near, 2)) .* dmin;
end
end

function s = segdist(Q, U, E)
% computes distance from every point Q to every line segment U -> U + E
% wx, wy, wz are the components of w_{ij} = p_i - U_j
% distance function f would be f(t) = |w-t*E|^2 and the minimum is at 
% t^* = w\cdot E/|E|^2
wx = Q(:,1) - U(:,1).';
wy = Q(:,2) - U(:,2).';
wz = Q(:,3) - U(:,3).';
% take only infimum over [0,1]
% left of it: 0
% right of it: 1
t  = min(max((wx.*E(:,1).' + wy.*E(:,2).' + wz.*E(:,3).') ./ sum(E.^2, 2).', 0), 1);
s  = sqrt((wx - t.*E(:,1).').^2 + (wy - t.*E(:,2).').^2 + (wz - t.*E(:,3).').^2);
% s is finally the distance
end

function n = outward_normals(pos, tri, c)
% unit triangle normals, pointing away from the star centre c
% get all triangles
A = pos(tri(:,1),:);
B = pos(tri(:,2),:);
C = pos(tri(:,3),:);
n = cross(B - A, C - A, 2);
n = n ./ sqrt(sum(n.^2,2));
flip = sum(n .* ((A + B + C)/3 - c), 2) < 0; % if the normal looks to the inside ... then flip
n(flip,:) = -n(flip,:);
end


function plot_segs(P1, P2, varargin)
% draw the segments P1(i,:) -> P2(i,:) as one line object
m = size(P1, 1);
X = [P1(:,1) P2(:,1) nan(m,1)].';
Y = [P1(:,2) P2(:,2) nan(m,1)].';
Z = [P1(:,3) P2(:,3) nan(m,1)].';
plot3(X(:), Y(:), Z(:), varargin{:});
end

function C = bwr(m)
% m colours from blue over white to red
t = linspace(-1, 1, m).';
C = [min(1, 1 + t), 1 - abs(t), min(1, 1 - t)];
end