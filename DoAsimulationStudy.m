%% Direction of Arrival (DoA) based spoof detection simulation
% 
% This script is to compare the performance of the likelihood ratio test
% based detection developped at Stanford an SVD based approach used e.g. in
% 
% Appel, M. et al. Experimental validation of GNSS repeater detection 
% based on antenna arrays for maritime applications. CEAS Sp. J. 11, 
% 719 (2019).
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

% For sequential approach: set number of measurements to consider
nSeq = 3;


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
[SigCA, logLambdaCA, logLambdaNormCAseq] = deal(zeros(1, K));
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
    
    % sequential Lambda (consider nSeq epochs)
    seqEp = max([1, k-nSeq+1]) : k;
    logLambdaNormCAseq(k) = sum(logLambdaCA(seqEp)) ...
        / sqrt(sum(SigCA(seqEp)));
        
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
p1 = plot(logLambdaCA ./ sqrt(SigCA), [linestyle(1), ':']);
p2 = plot(logLambdaNormCAseq, linestyle, 'LineWidth', 1.5);
% ylim([-25 5])
plot(fig1.CurrentAxes.YLim(2)*(logLambdaCA ./ sqrt(SigCA) < PhiT) ...
    + fig1.CurrentAxes.YLim(1)*(logLambdaCA ./ sqrt(SigCA) >= PhiT), ...
        'b', 'LineWidth', 1.5)
fig1.CurrentAxes.Children = flipud(fig1.CurrentAxes.Children);
line([0 K], [PhiT PhiT], 'Color', 'k') % decision threshold
% ylim(fig1.CurrentAxes.YLim)
text(K*0.75, PhiT-2, ['$P_{FA_{max}} = 10^{', num2str(pFAmaxPot), '}$'], ...
    'FontSize', fs-6, 'Interpreter', 'latex')
legend([p1; p2], {'snapshot'; 'sequential'}, ...
    'Location', 'southeast', 'FontSize', fs-8, 'Interpreter', 'latex')
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
%% functions

function q = euler2q(eulerAngles)
%euler2q(eulerAngles)
%   Calculates quaternions q from Euler angles roll, pitch, yaw. Returns
%   angles in a vector of dimensions similar to dimensions of eulerAngles.
%   Operates on each column of eulerAngles, unless the number of rows of 
%   eulerAngles is not 3.
%   Input:
%   eulerAngles array<double> of calculated Euler Angles. Can be 3 x N or 
%               N x 3.
%   
%   Output:
%   q           array<double> of quaternions. In dimensions
%               similar to q. That is, if q is a 4 x N matrix, then
%               eulerAngles is a 3 x N matrix and vice versa.
%   
%   Based on the formulas in "Representing Attitude: Euler Angles, Unit 
%   Quaternions and Rotation Vectors" by James Diebel, 20 Oct. 2006.
%   
%   Written by Fabian Rothmaier, 2019

% ensure working with column vectors of Euler angles
if size(eulerAngles, 1) ~= 3
    flipDim = true;
    seA = sin(eulerAngles' / 2);
    ceA = cos(eulerAngles' / 2);
else
    flipDim = false;
    seA = sin(eulerAngles / 2);
    ceA = cos(eulerAngles / 2);
end
q = [prod(ceA, 1) + prod(seA, 1); ...
     -ceA(1, :) .* seA(2, :) .* seA(3, :) + ...
        seA(1, :) .* ceA(2, :) .* ceA(3, :); ...
     ceA(1, :) .* seA(2, :) .* ceA(3, :) + ...
        seA(1, :) .* ceA(2, :) .* seA(3, :); ...
     ceA(1, :) .* ceA(2, :) .* seA(3, :) - ...
        seA(1, :) .* seA(2, :) .* ceA(3, :)];

% transpose again to original dimensions if necessary
if flipDim
    q = q';
end

end

function eulerAngles = q2euler(q)
%q2euler(q)
%   Calculates Euler angles roll, pitch, yaw from quaternions q. Returns
%   angles in a vector of dimensions similar to dimensions of q.
%   Operates on each column of q, unless the number of rows of q is not 4.
%   
%   Input:
%   q           array<double> of quaternions. Can be 4 x N or N x 4.
%   
%   Output:
%   eulerAngles array<double> of calculated Euler Angles. In dimensions
%               similar to q. That is, if q is a 4 x N matrix, then
%               eulerAngles is a 3 x N matrix and vice versa.
%   
%   Based on the formulas in "Representing Attitude: Euler Angles, Unit 
%   Quaternions and Rotation Vectors" by James Diebel, 20 Oct. 2006.
%   
%   Written by Fabian Rothmaier, 2019

% ensure working with column vectors of quaternions
if size(q, 1) ~= 4
    flipDim = true;
    q = q';
else
    flipDim = false;
end
eulerAngles = [atan2(2.*q(3, :).*q(4, :) + 2.*q(1, :).*q(2, :), ...
                     q(1, :).^2 + q(4, :).^2 - q(2, :).^2 - q(3, :).^2); ...
               -asin(2.*q(2, :).*q(4, :) - 2.*q(1, :).*q(3, :)); ...
               atan2(2.*q(2, :).*q(3, :) + 2.*q(1, :).*q(4, :), ...
                     q(1, :).^2 + q(2, :).^2 - q(3, :).^2 - q(4, :).^2)];

% transpose again to original dimensions if necessary
if flipDim
    eulerAngles = eulerAngles';
end

end


function r = qMult(p, q)
%qMult(p, q)
%   Quaternion multiplication p * q.
%   Follows the convention that multiplication is right to left.
%   
%   Expects p and q to have the same dimensions. They can be 4 x N or N x 4
%   arrays.
%   
%   Input:
%   p   array<double> of quaternions. Can be 4 x N or N x 4.
%   q   array<double> of quaternions. Must be same dimensions as p.
%   
%   Output:
%   r   array<double> of quaternions. Same dimensions as p.
%   
%   Based on the formulas in "Representing Attitude: Euler Angles, Unit 
%   Quaternions and Rotation Vectors" by James Diebel, 20 Oct. 2006.
%   
%   Written by Fabian Rothmaier, 2019

% ensure working with column vectors of quaternions
if size(q, 1) ~= 4
    flipDim = true;
    q = q';
    p = p';
else
    flipDim = false;
end

% generate rotation matrix
Q = @(p) [p(1) -p(2:4)'; 
          p(2) p(1) p(4) -p(3);
          p(3) -p(4) p(1) p(2);
          p(4) p(3) -p(2) p(1)];

% compute multiplication for each quaternion pair
N = size(q, 2);
r = zeros(4, N);
for n = 1:N
    r(:, n) = Q(p(:, n)) * q(:, n);
end

% revert dimensions back if necessary
if flipDim
    r = r';
end

end


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
