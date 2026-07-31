% baseline_test.m
% Implements MRAC and Dual IOP baselines for camera-ready manuscript integration.
%{
Filename:        baseline_test.m
Author:          Vittorio Lippi
Date:            2026-7-25
Version:         1.0
Purpose:
    Provide adaptive control experiments and comparison between a classic
    Model Reference Adaptive Controller (MRAC) and a DUAL IOP adaptive
    architecture. The script implements plant and reference-model
    definitions, controller parameter updates, simulation orchestration,
    and plotting of performance and adaptation metrics.

Scope:
    - Build and simulate nominal plant and reference model.
    - Implement classic MRAC (direct and/or indirect forms).
    - Implement DUAL IOP (dual-optimizer / dual-identifier) controller.
    - Compare tracking performance, parameter convergence, and robustness.

Usage:
    Configure parameters in the "User Settings" section, then run the
    script from the MATLAB command window or the Editor. Results include
    time-series plots of states, control input, tracking error, and
    parameter estimates.

Assumptions:
    - Plant and reference model are correctly specified and stabilizable.
    - Required toolboxes: Control System Toolbox (optional: Simulink if used).
    - Units and signals are consistent across definitions.

References:
    - Standard MRAC literature (e.g., Ioannou & Sun; Narendra & Annaswamy).
    - Internal notes describing the DUAL IOP adaptation strategy.
%}



clear; clc; close all;

% --- CONFIGURATION (Matched to Benchmark_exo.m) ---
A_nom = 1.05; B_true = 0.15; 
noise_std = 0.025; 
N_steps = 150;
demo_steps = 100; 

% Baseline PD Controller Gains
Kp = 0.60; Kd = 0.25;

% Trajectories
x_ref_calib = 0.3 * sin(2*pi * (1:N_steps+1) / 30);
x_ref_deploy = 1.0 * sin(2*pi * (1:demo_steps+1) / 30);

% Physical saturation to prevent infinite numerical explosions
phys_cap = @(val) sign(val) * min(abs(val), 5.0);

%{
Section: Controller Implementations
Description:
    This section contains two controller implementations used for
    comparison and study.

Classic MRAC:
    - Objective: Force the uncertain plant to follow a specified
      reference model using adaptive parameter updates derived from
      Lyapunov stability arguments.
    - Key elements:
        * Reference model Am, Bm defining desired closed-loop dynamics.
        * Adaptive law (direct or indirect) updating controller gains
          (e.g., kx, kr or estimated plant A_hat, B_hat) using the
          tracking error e(t) = x(t) - x_m(t) and a positive-definite
          Lyapunov matrix P.
        * Optional robustness modifications: sigma-modification,
          e-modification, projection, parameter dead-zone.
    - Typical update form (direct MRAC):
        kx_dot = Gamma_x * x(t) * e(t)' * P * B + sigma * kx
        kr_dot = Gamma_r * r(t) * e(t)' * P * B + sigma * kr
    - Notes:
        * Tune adaptation gains Gamma_* to trade off speed vs. robustness.
        * Ensure persistency of excitation for parameter convergence.

DUAL IOP:
    - Objective: Combine an inner identification/optimization loop with
      an outer control loop to improve adaptation under uncertainty and
      disturbances by explicitly optimizing identification and control
      objectives (Dual Identification-Optimization Process).
    - Key elements:
        * Dual structure: one pathway estimates model/uncertainty
          (identifier), another computes control using the current
          estimate (optimizer/controller).
        * Identifier may use batch/recursive least squares, RBF
          disturbance model, or a learning-based estimator.
        * Optimizer updates control parameters to minimize a
          performance cost (tracking + control effort) using the latest
          model estimate; may include momentum, regularization, or
          constraints.
        * Interaction: identifier provides improved model for optimizer;
          optimizer provides probing/inputs that aid identification.
    - Typical update behavior:
        * Identifier: Theta_dot = Gamma_id * phi(x,u) * e' * P_id + reg
        * Optimizer: minimize J(u; Theta) or update control gains via
          gradient-like rule using estimated gradients and momentum.
    - Notes:
        * DUAL IOP can improve transient performance and robustness but
          requires careful coordination (learning rates, regularization).
        * Monitoring stability (Lyapunov or input-to-state) is important,
          especially when optimizer introduces noncausal or aggressive updates.
%}


fprintf('\n=== RUNNING BASELINE A: MRAC (COMPOSITE ADAPTIVE) ===\n');
% Reference Model: xm_k+1 = am * xm_k + bm * r_k
am = 0.5; bm = 0.5; 
gamma_mrac = 0.05; % Normalized gradient descent learning rate

x_mrac = zeros(1, demo_steps+1);
xm = zeros(1, demo_steps+1);
theta_x = 0; theta_r = 0; % Initial adaptive parameters

for k = 1:demo_steps
    w_k = noise_std * randn();
    rk = x_ref_deploy(k);
    
    % Reference model forward step
    xm(k+1) = am * xm(k) + bm * rk;
    
    % 1. Stabilizing baseline PD control
    if k==1, vel1=0; else, vel1=x_mrac(k)-x_mrac(k-1); end
    if k==1, vref=0; else, vref=x_ref_deploy(k)-x_ref_deploy(k-1); end
    
    u_ff = x_ref_deploy(k+1) - A_nom * x_ref_deploy(k);
    u_base = u_ff - Kp*(x_mrac(k) - x_ref_deploy(k)) - Kd*(vel1 - vref);
    
    % 2. Linear-in-the-parameters (LIP) MRAC acting as compensator
    u_mrac_ad = theta_x * x_mrac(k) + theta_r * rk;
    u_total = u_base + u_mrac_ad;
    
    % 3. Plant dynamics
    x_mrac(k+1) = phys_cap(A_nom * x_mrac(k) + B_true * x_mrac(k)^3 + u_total + w_k);
    
    % 4. Tracking error w.r.t reference model
    e_k = x_mrac(k+1) - xm(k+1);
    
    % 5. Parameter update via normalized gradient descent
    norm_factor = 1 + x_mrac(k)^2 + rk^2;
    theta_x = theta_x - gamma_mrac * e_k * x_mrac(k) / norm_factor;
    theta_r = theta_r - gamma_mrac * e_k * rk / norm_factor;
end


fprintf('=== RUNNING BASELINE B: DUAL IOP ===\n');
% PHASE 1: Calibration & Linear Least Squares
x_cal = zeros(1, N_steps+1);
u_cal = zeros(1, N_steps);

for k = 1:N_steps
    w_k = noise_std * randn();
    if k == 1, vel = 0; vel_ref = 0; else
        vel = x_cal(k) - x_cal(k-1); vel_ref = x_ref_calib(k) - x_ref_calib(k-1);
    end
    u_ff = x_ref_calib(k+1) - A_nom * x_ref_calib(k);
    u_base = u_ff - Kp * (x_cal(k) - x_ref_calib(k)) - Kd * (vel - vel_ref);
    
    u_cal(k) = u_base;
    x_cal(k+1) = A_nom * x_cal(k) + B_true * x_cal(k)^3 + u_base + w_k;
end

% Construct Dual IOP residual signals and solve via Least Squares (Phase 1)
r_obs = x_cal(2:end) - (A_nom * x_cal(1:end-1) + u_cal);
Phi_mat = x_cal(1:end-1)'; 
Q_est = pinv(Phi_mat) * r_obs'; 

% PHASE 2: Static Deployment on Large Reference
x_iop = zeros(1, demo_steps+1);
for k = 1:demo_steps
    w_k = noise_std * randn();
    if k==1, vel=0; else, vel=x_iop(k)-x_iop(k-1); end
    if k==1, vref=0; else, vref=x_ref_deploy(k)-x_ref_deploy(k-1); end
    
    u_ff = x_ref_deploy(k+1) - A_nom * x_ref_deploy(k);
    u_base = u_ff - Kp*(x_iop(k) - x_ref_deploy(k)) - Kd*(vel - vref);
    
    % Deploy frozen Q parameter
    u_iop_corr = Q_est * x_iop(k);
    u_tot = u_base - u_iop_corr;
    
    x_iop(k+1) = phys_cap(A_nom * x_iop(k) + B_true * x_iop(k)^3 + u_tot + w_k);
end

fprintf('\n=== QUANTITATIVE PERFORMANCE METRICS ===\n');

% Compute Tracking MSE over the full deployment window
mse_mrac = mean((x_mrac(1:demo_steps) - x_ref_deploy(1:demo_steps)).^2);
mse_iop  = mean((x_iop(1:demo_steps) - x_ref_deploy(1:demo_steps)).^2);

% --- Calculate Amplitude/RMS Error (Steady-State) ---
% Evaluate the second half of deployment to allow initial transients to settle
eval_window = floor(demo_steps/2):demo_steps;
ref_eval = x_ref_deploy(eval_window);
mrac_eval = x_mrac(eval_window);
iop_eval = x_iop(eval_window);

rms_ref = rms(ref_eval);
rms_mrac = rms(mrac_eval);
rms_iop = rms(iop_eval);

amp_error_percent_mrac = abs(rms_ref - rms_mrac) / rms_ref * 100;
amp_error_percent_iop = abs(rms_ref - rms_iop) / rms_ref * 100;

fprintf('MRAC STEADY-STATE ANALYSIS:\n');
fprintf(' - Tracking MSE: %.4f\n', mse_mrac);
fprintf(' - Amplitude Discrepancy (RMS): %.2f%%\n', amp_error_percent_mrac);

fprintf('\nDUAL IOP STEADY-STATE ANALYSIS:\n');
fprintf(' - Tracking MSE: %.4f\n', mse_iop);
fprintf(' - Amplitude Discrepancy (RMS): %.2f%%\n', amp_error_percent_iop);

% 1. Generate Figure
fig = figure('Name', 'Classical & Structural Baselines', 'Position', [200, 200, 800, 500]);

plot(0:demo_steps, x_ref_deploy, 'k:', 'LineWidth', 2, 'DisplayName', 'Reference Trajectory'); hold on; grid on;

% FIX: Added a LaTeX non-breaking space (~) at the end of the display names.
% This forces MATLAB to calculate a wider bounding box, preventing frame overflow.
plot(0:demo_steps, x_mrac, 'b-', 'LineWidth', 1.5, 'DisplayName', sprintf('MRAC (Amp. Error: %.2f\\%%)~', amp_error_percent_mrac));
plot(0:demo_steps, x_iop, 'r-', 'LineWidth', 1.5, 'DisplayName', sprintf('Dual IOP (Amp. Error: %.2f\\%%)~', amp_error_percent_iop));

xlabel('Time Step $k$', 'Interpreter', 'latex');
ylabel('State $x_k$ (rad)', 'Interpreter', 'latex');
title('Evaluation of Proposed Baselines on 1-DOF Nonlinear System', 'Interpreter', 'latex', 'FontSize', 14);

% Generate legend
legend('Location', 'southwest', 'Interpreter', 'latex');
ylim([-2.0, 2.0]);
% Alternative fix: Remove the bounding box outline entirely
legend('Location', 'southwest', 'Interpreter', 'latex', 'Box', 'off');
% Force the 'painters' renderer for flawless LaTeX PDF export
set(fig, 'Renderer', 'painters');
ax = gca;                       % current axes
ax.GridColor = [0.9 0.9 0.9]; % light gray (RGB)
ax.GridAlpha = 0.9;             % opacity (0 = transparent, 1 = solid)
ax.LineWidth = 0.5;             % optional: thinner grid lines
grid(ax, 'on');

% Uncomment below to automatically export a high-quality PDF 
exportgraphics(fig, 'Peer_Review_Baselines_Final.pdf', 'ContentType', 'vector');

% 2. Generate LaTeX Table
fprintf('\n%% --- Copy this table into your manuscript ---\n');
fprintf('\\begin{table}[htpb]\n');
fprintf('\\centering\n');
fprintf('\\caption{Empirical Tracking Error of Structural and Adaptive Baselines}\n');
fprintf('\\label{tab:reviewer_baselines}\n');
fprintf('\\begin{tabular}{lcc}\n');
fprintf('\\hline\n');
fprintf('\\textbf{Architecture} & \\textbf{Tracking Error (MSE)} & \\textbf{Amplitude Distortion (\\%%)} \\\\\n');
fprintf('\\hline\n');
fprintf('MRAC (Linear-in-the-Parameters) & %.4f & %.2f \\\\\n', mse_mrac, amp_error_percent_mrac);
fprintf('Dual IOP (Frozen Phase 2)       & %.4f & %.2f \\\\\n', mse_iop, amp_error_percent_iop);
fprintf('\\hline\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n\n');