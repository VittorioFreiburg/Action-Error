% test_benchmark_script_final_master.m
%clear; %clc; %close all;

% --- CONFIGURATION ---
A_nom = 1.05; B_true = 0.15; % Duffing Exoskeleton Joint
noise_std = 0.025; % Increased noise for realistic jitter
N_steps = 150;
demo_steps = 100; % Full limit cycles

% PD Controller Gains
Kp = 0.60; Kd = 0.25;

%% --- PHASE 1: CALIBRATION (SMALL AMPLITUDE) ---
fprintf('=== PHASE 1: COLLECTING CALIBRATION DATA ===\n');
x = zeros(1, N_steps+1); x(1) = 0;
u_total_history = zeros(1, N_steps);

x_ref_calib = 0.3 * sin(2*pi * (1:N_steps+1) / 30);

for k = 1:N_steps
    w_k = noise_std * randn();
    if k == 1, vel = 0; vel_ref = 0; else
        vel = x(k) - x(k-1); vel_ref = x_ref_calib(k) - x_ref_calib(k-1);
    end

    u_ff = x_ref_calib(k+1) - A_nom * x_ref_calib(k);
    u_base = u_ff - Kp * (x(k) - x_ref_calib(k)) - Kd * (vel - vel_ref);

    u_total_history(k) = u_base;
    x(k+1) = A_nom * x(k) + B_true * x(k)^3 + u_base + w_k;
end

%% --- PHASE 2: OFFLINE NN TRAINING ---
fprintf('=== PHASE 2: OFFLINE NN TRAINING ===\n');
x_k = x(1:end-1)'; x_next = x(2:end)';
u_total = u_total_history';

r = x_next - (A_nom * x_k + u_total);
inputs = x_k'; targets = r';

mu_in = mean(inputs, 2); sigma_in = std(inputs, 0, 2); sigma_in(sigma_in==0) = 1;
inputs_n = (inputs - mu_in) ./ sigma_in;
mu_t = mean(targets, 2); sigma_t = std(targets, 0, 2); if sigma_t==0, sigma_t = 1; end
targets_n = (targets - mu_t) ./ sigma_t;

hidden_size = 8;
net = feedforwardnet(hidden_size, 'trainlm');
net.divideParam.trainRatio = 0.8; net.divideParam.valRatio = 0.1; net.divideParam.testRatio = 0.1;
net.trainParam.epochs = 200; net.trainParam.showWindow = false;
[net, ~] = train(net, inputs_n, targets_n);

nn_predict = @(xk) (sigma_t .* net((xk - mu_in)./sigma_in) + mu_t);

%% --- PHASE 3: PARALLEL EVALUATION (LARGE AMPLITUDE) ---
fprintf('=== PHASE 3: EVALUATION & ABLATION STUDIES ===\n');

x_ref_deploy = 1.0 * sin(2*pi * (1:demo_steps+1) / 30);

x_base = zeros(1, demo_steps+1); x_off = zeros(1, demo_steps+1);
x_on   = zeros(1, demo_steps+1); x_ab1 = zeros(1, demo_steps+1);
x_ab2  = zeros(1, demo_steps+1); x_ab3 = zeros(1, demo_steps+1);

W1_init = net.IW{1,1}; b1_init = net.b{1}; W2_init = net.LW{2,1}; b2_init = net.b{2};
buf_in_init  = inputs(:, end-39:end); buf_tar_init = targets(:, end-39:end);

phys_cap = @(val) sign(val) * min(abs(val), 5.0);

% --- RUN 1 & 2: Baseline & Static Offline ---
for k = 1:demo_steps
    w_k = noise_std * randn();
    u_ff = x_ref_deploy(k+1) - A_nom * x_ref_deploy(k);
    if k==1, vref=0; else, vref=x_ref_deploy(k)-x_ref_deploy(k-1); end

    % 1. Baseline
    if k==1, vel1=0; else, vel1=x_base(k)-x_base(k-1); end
    u_base1 = u_ff - Kp*(x_base(k) - x_ref_deploy(k)) - Kd*(vel1 - vref);
    x_base(k+1) = A_nom*x_base(k) + B_true*x_base(k)^3 + u_base1 + w_k;

    % 2. Offline NN
    if k==1, vel2=0; else, vel2=x_off(k)-x_off(k-1); end
    u_base2 = u_ff - Kp*(x_off(k) - x_ref_deploy(k)) - Kd*(vel2 - vref);
    u_total2 = u_base2 - nn_predict(x_off(k));
    x_off(k+1) = phys_cap(A_nom*x_off(k) + B_true*x_off(k)^3 + u_total2 + w_k);
end

% --- RUN 3, 4, 5, 6: Online Algorithms ---
lr_optimal  = 0.50;  % The new fast baseline
lr_sluggish = 0.01;  % Too slow (Tracking error)
lr_unstable = 0.50;  % Way too fast + No Clip (Explodes)

W1_on = W1_init; b1_on = b1_init; W2_on = W2_init; b2_on = b2_init; buf_in_on = buf_in_init; buf_tar_on = buf_tar_init;
W1_a1 = W1_init; b1_a1 = b1_init; W2_a1 = W2_init; b2_a1 = b2_init;
W1_a2 = W1_init; b1_a2 = b1_init; W2_a2 = W2_init; b2_a2 = b2_init; buf_in_a2 = buf_in_init; buf_tar_a2 = buf_tar_init;
W1_a3 = W1_init; b1_a3 = b1_init; W2_a3 = W2_init; b2_a3 = b2_init; buf_in_a3 = buf_in_init; buf_tar_a3 = buf_tar_init;

% Initialize history arrays for ALL online architectures
W1_on_hist = zeros(size(W1_on, 1), size(W1_on, 2), demo_steps);
W2_on_hist = zeros(size(W2_on, 1), size(W2_on, 2), demo_steps);

W1_a1_hist = zeros(size(W1_a1, 1), size(W1_a1, 2), demo_steps);
W2_a1_hist = zeros(size(W2_a1, 1), size(W2_a1, 2), demo_steps);

W1_a2_hist = zeros(size(W1_a2, 1), size(W1_a2, 2), demo_steps);
W2_a2_hist = zeros(size(W2_a2, 1), size(W2_a2, 2), demo_steps);

W1_a3_hist = zeros(size(W1_a3, 1), size(W1_a3, 2), demo_steps);
W2_a3_hist = zeros(size(W2_a3, 1), size(W2_a3, 2), demo_steps);

%Run!

for k = 1:demo_steps
    w_k = noise_std * randn();
    u_ff = x_ref_deploy(k+1) - A_nom * x_ref_deploy(k);
    if k==1, vref=0; else, vref=x_ref_deploy(k)-x_ref_deploy(k-1); end

    % ==========================================================
    % RUN 3: Optimal Online (Fast LR, Gradient Clip, NO BUFFER)
    % ==========================================================
    if k==1, vel=0; else, vel=x_on(k)-x_on(k-1); end
    u_base = u_ff - Kp*(x_on(k) - x_ref_deploy(k)) - Kd*(vel - vref);
    
    % Forward Pass (Single State)
    in_n = (x_on(k) - mu_in)/sigma_in; 
    a1 = tanh(W1_on*in_n + b1_on);
    u_nn = (W2_on*a1 + b2_on)*sigma_t + mu_t;
    
    % Plant Step
    x_on(k+1) = phys_cap(A_nom*x_on(k) + B_true*x_on(k)^3 + (u_base - u_nn) + w_k);
    r_obs = x_on(k+1) - (A_nom*x_on(k) + u_base - u_nn);
    
    % --- Pure SGD Backpropagation (No Buffer) ---
    TAR_n = (r_obs - mu_t)/sigma_t; 
    E_n = (W2_on*a1 + b2_on) - TAR_n;
    
    dW2 = E_n * a1'; 
    db2 = E_n;
    dZ1 = (W2_on'*E_n) .* (1 - a1.^2); 
    dW1 = dZ1 * in_n'; 
    db1 = dZ1;
    
    % --- Safety Filter: Gradient Clipping ---
    if norm(dW2,'fro')>0.5, dW2=dW2*(0.5/norm(dW2,'fro')); end
    if norm(dW1,'fro')>0.5, dW1=dW1*(0.5/norm(dW1,'fro')); end
    
    % Weight Update
    W1_on = W1_on - lr_optimal*dW1; 
    b1_on = b1_on - lr_optimal*db1; 
    W2_on = W2_on - lr_optimal*dW2; 
    b2_on = b2_on - lr_optimal*db2;

    % % ==========================================================
    % % RUN 4: Ablation 1 (No Buffer - Catastrophic Forgetting)
    % % ==========================================================
    % if k==1, vel=0; else, vel=x_ab1(k)-x_ab1(k-1); end
    % u_base = u_ff - Kp*(x_ab1(k) - x_ref_deploy(k)) - Kd*(vel - vref);
    % in_n = (x_ab1(k) - mu_in)/sigma_in; a1 = tanh(W1_a1*in_n + b1_a1);
    % u_nn = (W2_a1*a1 + b2_a1)*sigma_t + mu_t;
    % 
    % x_ab1(k+1) = phys_cap(A_nom*x_ab1(k) + B_true*x_ab1(k)^3 + (u_base - u_nn) + w_k);
    % r_obs = x_ab1(k+1) - (A_nom*x_ab1(k) + u_base - u_nn);
    % 
    % TAR_n = (r_obs - mu_t)/sigma_t; E_n = (W2_a1*a1 + b2_a1) - TAR_n;
    % dW2 = E_n*a1'; db2 = E_n; dZ1 = (W2_a1'*E_n).*(1-a1.^2); dW1 = dZ1*in_n'; db1 = dZ1;
    % 
    % if norm(dW2,'fro')>0.5, dW2=dW2*(0.5/norm(dW2,'fro')); end
    % if norm(dW1,'fro')>0.5, dW1=dW1*(0.5/norm(dW1,'fro')); end
    % W1_a1 = W1_a1 - lr_optimal*dW1; b1_a1 = b1_a1 - lr_optimal*db1; W2_a1 = W2_a1 - lr_optimal*dW2; b2_a1 = b2_a1 - lr_optimal*db2;

    % ==========================================================
    % RUN 5: Ablation 2 (Sluggish Learning - Lag)
    % ==========================================================
    if k==1, vel=0; else, vel=x_ab2(k)-x_ab2(k-1); end
    u_base = u_ff - Kp*(x_ab2(k) - x_ref_deploy(k)) - Kd*(vel - vref);
    in_n = (x_ab2(k) - mu_in)/sigma_in; a1 = tanh(W1_a2*in_n + b1_a2);
    u_nn = (W2_a2*a1 + b2_a2)*sigma_t + mu_t;

    x_ab2(k+1) = phys_cap(A_nom*x_ab2(k) + B_true*x_ab2(k)^3 + (u_base - u_nn) + w_k);
    r_obs = x_ab2(k+1) - (A_nom*x_ab2(k) + u_base - u_nn);

    buf_in_a2 = [buf_in_a2(2:end), x_ab2(k)]; buf_tar_a2 = [buf_tar_a2(2:end), r_obs];
    IN_n = (buf_in_a2 - mu_in)/sigma_in; TAR_n = (buf_tar_a2 - mu_t)/sigma_t; N_b = length(TAR_n);
    A1 = tanh(W1_a2*IN_n + b1_a2); E_n = (W2_a2*A1 + b2_a2) - TAR_n;
    dW2 = (E_n*A1')/N_b; db2 = sum(E_n, 2)/N_b;
    dZ1 = (W2_a2'*E_n).*(1-A1.^2); dW1 = (dZ1*IN_n')/N_b; db1 = sum(dZ1, 2)/N_b;

    if norm(dW2,'fro')>0.5, dW2=dW2*(0.5/norm(dW2,'fro')); end
    if norm(dW1,'fro')>0.5, dW1=dW1*(0.5/norm(dW1,'fro')); end
    W1_a2 = W1_a2 - lr_sluggish*dW1; b1_a2 = b1_a2 - lr_sluggish*db1; W2_a2 = W2_a2 - lr_sluggish*dW2; b2_a2 = b2_a2 - lr_sluggish*db2;

    % ==========================================================
    % RUN 6: Ablation 3 (Violating Time-Scale Separation - Unstable)
    % ==========================================================
    if k==1, vel=0; else, vel=x_ab3(k)-x_ab3(k-1); end
    u_base = u_ff - Kp*(x_ab3(k) - x_ref_deploy(k)) - Kd*(vel - vref);
    in_n = (x_ab3(k) - mu_in)/sigma_in; a1 = tanh(W1_a3*in_n + b1_a3);
    u_nn = (W2_a3*a1 + b2_a3)*sigma_t + mu_t;

    x_ab3(k+1) = phys_cap(A_nom*x_ab3(k) + B_true*x_ab3(k)^3 + (u_base - u_nn) + w_k);
    r_obs = x_ab3(k+1) - (A_nom*x_ab3(k) + u_base - u_nn);

    buf_in_a3 = [buf_in_a3(2:end), x_ab3(k)]; buf_tar_a3 = [buf_tar_a3(2:end), r_obs];
    IN_n = (buf_in_a3 - mu_in)/sigma_in; TAR_n = (buf_tar_a3 - mu_t)/sigma_t; N_b = length(TAR_n);
    A1 = tanh(W1_a3*IN_n + b1_a3); E_n = (W2_a3*A1 + b2_a3) - TAR_n;
    dW2 = (E_n*A1')/N_b; db2 = sum(E_n, 2)/N_b;
    dZ1 = (W2_a3'*E_n).*(1-A1.^2); dW1 = (dZ1*IN_n')/N_b; db1 = sum(dZ1, 2)/N_b;

    % NO CLIPPING - Allow it to explode
    W1_a3 = W1_a3 - lr_unstable*dW1; b1_a3 = b1_a3 - lr_unstable*db1; W2_a3 = W2_a3 - lr_unstable*dW2; b2_a3 = b2_a3 - lr_unstable*db2;

    % Save weights at the end of step k for Tracking Error analysis
    W1_on_hist(:,:,k) = W1_on; W2_on_hist(:,:,k) = W2_on;
    W1_a1_hist(:,:,k) = W1_a1; W2_a1_hist(:,:,k) = W2_a1;
    W1_a2_hist(:,:,k) = W1_a2; W2_a2_hist(:,:,k) = W2_a2;
    W1_a3_hist(:,:,k) = W1_a3; W2_a3_hist(:,:,k) = W2_a3;

end


%% --- PLOTTING RESULTS (2x2 GRID) ---
fprintf('=== GENERATING PLOTS ===\n');
%figure;

c_ref = [0.6 0.6 0.6]; c_base = 'k'; c_off = 'r'; c_on = 'b';
c_ab1 = '#A0522D'; % Brown
c_ab2 = '#228B22'; % Green
c_ab3 = '#800080'; % Purple

% --- (a) Time Series: Standard Architecture ---
%subplot; hold on; grid on;
%plot;
%plot;
%plot;
%plot;
%xlabel; %ylabel;
%title;
%legend; %ylim;

% --- (b) Time Series: Online Ablations ---
%subplot; hold on; grid on;
%plot;
%plot;
%%plot;
%plot;
%plot;
%xlabel; %ylabel;
%title;
%legend; %ylim;

% --- (c) Return Map: Standard Architecture ---
%subplot; hold on; grid on;
%plot;
%plot;
%plot;
%xlabel; %ylabel;
%title;
%xline; %yline;
%xlim; %ylim;

% --- (d) Return Map: Online Ablations ---
%subplot; hold on; grid on;
%plot;
%%plot;
%plot;
%plot;
%xlabel; %ylabel;
%title;
%xline; %yline;
%xlim; %ylim;