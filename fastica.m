% Number of sources
num_sources = 3;

% Create random sources
signals = randn(num_sources, 1000);

% Mixing matrix
mixing_matrix = randn(num_sources);

% Mixed signals
mixed_signals = mixing_matrix * sources;

% initialize variables
W = eye(num_sources);
max_iter = 1000;
learning_rate = 0.1;

% centering the data
mean_sig = mean(signals, 2);
centered_signals = signals - mean_sig;

% whiten the data
[E, D] = eig(cov(centered_signals'));
whitening_matrix = E * diag(sqrt(1./(diag(D) + eps))) * E';
whitened_signals = whitening_matrix * centered_signals;

% perform fastICA
for iter = 1:max_iter
    % compute g(W^T * x)
    y = W' * whitened_signals;
    g = tanh(y);
    
    % compute g' (derivative of g)
    dg = 1 - g.^2;
    
    % update W
    dW = learning_rate * (whitened_signals * g' / size(whitened_signals, 2) - sum(diag(dg) * W, 1));
    W = W + dW;
    
    % orthogonalize W
    [U, S, V] = svd(W);
    W = U * V';
end

% unmix the signals
unmixed_signals = W' * whitened_signals;

% compare original and unmixed signals
figure;
subplot(2,1,1); plot(signals');
title('Original Signals');
subplot(2,1,2); plot(unmixed_signals');
title('Unmixed Signals');
