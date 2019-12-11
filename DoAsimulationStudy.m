%% Direction of Arrival (DoA) based spoof detection simulation
% 
% This script is to compare the performance of the likelihood ratio test
% based detection developped at Stanford an SVD based approach used e.g. in
% 
% Appel, M. et al. Experimental validation of GNSS repeater detection 
% based on antenna arrays for maritime applications. CEAS Sp. J. 11, 
% 7–19 (2019).
% 
% 
% When using code from this script please cite:
% 
% 
% For questions contact Fabian Rothmaier, fabianr@stanford.edu

%% prepare work environment

% clear workspace
close all;
clear variables;

% seed random number generator for repeatable results (for
% reproduceability)
rng(2);


%% user inputs
% choose scenario (1 => 15 deg^2 cov, 2 => 25 deg^2 cov)
scenario = 1;

% set false alert probability
pFAmaxPot = -8; % 10^x

% constellation parameters
N = 3; % number of satellites
azSpoof = 2; % azimuth direction of spoofer
elSpoof = 0.8; % elevation of spoofer


%% describe constellation for simulation

% set scenario parameters
if scenario == 1
    linestyle = 'g';
    K = 2000; % number of simulation runs
    spoofedEpochs = [200:400, 650:800, 1000:1500];
    covDeg = 15;
else
    linestyle = 'b+-';
    K = 3000; % number of runs for Monte-Carlo
    spoofedEpochs = [200:400, 650:800, 1000:2000];
    covDeg = 25;
end

azTrue = repmat(2 * pi * rand(N, 1), 1, K); % uniform random between 0 and 2 pi
elRand = repmat(pi / 2 * rand(N, 1), 1, K); % uniform random between 0 and pi
% azTrue = repmat(2 * pi * [0; 0.02], 1, K); % specific between 0 and 2 pi
% elRand = repmat(pi / 2 * [0.5; 0.5], 1, K); % specific between 0 and pi
% azTrue = 2*pi * rand(N, K); % random at each epoch
% elRand = pi/2 * rand(N, K); % random at each epoch
elTrue = max(min(elRand, (pi/2-1e-4)*ones(N, K)), zeros(N, K));


% generate measurement standard deviation
sigma = repmat(sqrt(covDeg)*pi/180, N, K) ;


% add spoofer in desired time slots
spoEps = false(1, K); spoEps(spoofedEpochs) = true;
azTrueS = azTrue; azTrueS(:, spoEps) = azSpoof;
elTrueS = elTrue; elTrueS(:, spoEps) = elSpoof;

%% add noise to generate measurements

[elMeas, azMeas] = deal(zeros(N, K));
for n = 1:N
    % convert true direction to quaternions
    qTrue = euler2q([zeros(1, K); elTrueS(n, :); azTrueS(n, :)]);
    
    % generate randomized noise quaternion
    alpha = sigma(n, :) .* randn(1, K); % magnitude of error
    theta = 2*pi * rand(1, K); % direction of error
    qNoise = [cos(alpha/2); ...
              sin(alpha/2) .* [zeros(1, K); sin(theta); cos(theta)]];
    
    % compute measurement quaternion
    qMeas = qMult(qNoise, qTrue);
    
    % convert back to euler angles
    eulerMeas = q2euler(qMeas);
    
    % extract az, el
    elMeas(n, :) = eulerMeas(2, :);
    azMeas(n, :) = mod(eulerMeas(3, :), 2*pi);
end

% adjust for +90 el jump
azMeas(elMeas > pi/2) = mod(azMeas(elMeas > pi/2) + pi, 2*pi);
elMeas(elMeas > pi/2) = pi - elMeas(elMeas > pi/2);

%% dimensionality reduction step
% compute \bar{y} and \bar{\phi} (2N-3 true and measured central angles)

[sM1, sM2] = deal(zeros(2*N-3, N));
sM1(1, 1) = 1; sM2(1, 2) = 1; 
for n = 3:N
    sM1(2*n-4:2*n-3, :) = [zeros(2, n-3), eye(2), zeros(2, N-(n-1))];
    sM2(2*n-4:2*n-3, :) = [zeros(2, n-1), ones(2, 1), zeros(2, N-n)];
end

phi_bar = getCentralAngle(sM1 * azTrue, sM2 * azTrue, ...
                          sM1 * elTrue, sM2 * elTrue);
y_bar = getCentralAngle(sM1 * azMeas, sM2 * azMeas, ...
                        sM1 * elMeas, sM2 * elMeas);

%% prepare simulation
% precompute quartile value
PhiT = norminv(10^pFAmaxPot);

% preallocate result variables
[SigCA, logLambdaCA] = deal(zeros(1, K));
Q = zeros(3, K);


%% run simulation
for k = 1:K
    
    % calculate SVD based metric
    
    B = [cos(azMeas(:, k)').*cos(elMeas(:, k)'); ...
         sin(azMeas(:, k)').*cos(elMeas(:, k)'); ...
         sin(elMeas(:, k)')];
    % rotate arbitrarily (does not change the result)
%     att = [(rand(1, 2)-0.5)*pi, rand(1)*2*pi];
%     R1 = [1 0 0; ...
%           0 cos(att(1)) sin(att(1)); ...
%           0 -sin(att(1)) cos(att(1))];
%     R2 = [cos(att(2)) 0 -sin(att(2)); ...
%           0 1 0; ...
%           sin(att(2)) 0 cos(att(2))];
%     R3 = [cos(att(3)) sin(att(3)) 0; ...
%           -sin(att(3)) cos(att(3)) 0; ...
%           0 0 1];
    
    A = [cos(azTrue(:, 1)').*cos(elTrue(:, 1)'); ...
         sin(azTrue(:, 1)').*cos(elTrue(:, 1)'); ...
         sin(elTrue(:, 1)')];
    Q(:, k) = svd(B*A') / N; % normalized singular values
     
    % central angle measurement covariance (this overbounds the true error)
    S_barCA = diag((sM1+sM2)*sigma(:, k).^2);
    
    
    %% compute normalized log Lambda
    
    % Variance
    SigCA(k) = phi_bar(:, k)' * (S_barCA \ phi_bar(:, k));
    
    % log(Lambda(y)) distance from its mean under H_0
    logLambdaCA(k) = phi_bar(:, k)' * (S_barCA \ (y_bar(:, k) - phi_bar(:, k)));
    
end

%% Plot results

fs = 20; % figure font size

% histogram of normalized nominal log Lambdas
fH = figure; hold on; grid on;
histogram(logLambdaCA(~spoEps)./sqrt(SigCA(~spoEps)), min(round(K/50), 200), ...
    'normalization', 'pdf', ...
    'FaceColor', 'g', 'EdgeColor', 'none');
plot(linspace(PhiT, -PhiT), normpdf(linspace(PhiT, -PhiT)), ...
    'Color', [0 0.5 0], 'LineWidth', 1.5)
line(PhiT*ones(1, 2), fH.CurrentAxes.YLim, 'Color', 'k', 'LineWidth', 2)
xlabel('$\log\Lambda(\bar{y}) | H_0$ normalized', ...
    'FontSize', fs, 'Interpreter', 'latex')
ylabel('pdf', 'FontSize', fs, 'Interpreter', 'latex')

% snapshot normalized log Lambdas using great circle arcs
fig1 = figure; hold on; grid on;
plot(logLambdaCA ./ sqrt(SigCA), linestyle, 'LineWidth', 1.5);
% ylim([-25 5])
plot(fig1.CurrentAxes.YLim(2)*(logLambdaCA ./ sqrt(SigCA) < PhiT) ...
    + fig1.CurrentAxes.YLim(1)*(logLambdaCA ./ sqrt(SigCA) >= PhiT), ...
        'b', 'LineWidth', 1.5)
fig1.CurrentAxes.Children = flipud(fig1.CurrentAxes.Children);
line([0 K], [PhiT PhiT], 'Color', 'k') % decision threshold
% ylim(fig1.CurrentAxes.YLim)
text(K*0.75, PhiT-2, ['$P_{FA_{max}} = 10^{', num2str(pFAmaxPot), '}$'], ...
    'FontSize', fs-6, 'Interpreter', 'latex')
title(['$\sigma^2 = ', num2str(covDeg), '$ deg$^2$, ', ...
    'snapshot-based'], ...
    'FontSize', fs, 'Interpreter', 'latex')
xlabel('Simulation run', 'FontSize', fs, 'Interpreter', 'latex')
ylabel('Normalized $\log\Lambda(\bar{y}_t)$', ...
    'FontSize', fs, 'Interpreter', 'latex')

% to compare: normalized sum of singular values
figure; hold on; grid on;
plot(sum(Q), linestyle, 'LineWidth', 1.5)
ylim([0 1.1])
xlabel('Simulation run', 'FontSize', fs, 'Interpreter', 'latex')
ylabel('Normalized $\sum\sigma_i$', ...
    'FontSize', fs, 'Interpreter', 'latex')


function centralAngle = getCentralAngle(lambda1, lambda2, phi1, phi2)
%centralAngle = getCentralAngle(phi1, phi2)
%   Computes the central angle between two points on a sphere, defined by
%   azimuths lambda1, lambda2 and elevations phi1 and phi2.
%   All four passed angles must be arrays of equal size.

dL = lambda1 - lambda2;
sP1 = sin(phi1); cP1 = cos(phi1);
sP2 = sin(phi2); cP2 = cos(phi2);
centralAngle = atan2( sqrt( (cP2.*sin(dL)).^2 ...
                            + (cP1.*sP2 - sP1.*cP2.*cos(dL)).^2 ), ...
                    sP1.*sP2 + cP1.*cP2.*cos(dL) );

end
