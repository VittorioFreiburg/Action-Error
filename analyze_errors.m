% Analyze_Errors.m
% Run this AFTER Benchmark_exo.m
fprintf('\n=== QUANTIFYING CLOSED-LOOP IDENTIFICATION ERROR ===\n');

% Extract the calibration data from the workspace
x_calib = x(1:N_steps); 
u_base_calib = u_total_history(1:N_steps);

% Compute the Identification Errors (Exact, Proxy, and Ghost Variance)
[exact_id_error, estimated_bias, expected_bias_mse] = compute_identification_error(...
    nn_predict, x_calib, u_base_calib, x_ref_calib, B_true);

fprintf('Exact Functional Identification Error (MSE): %.6f\n', exact_id_error);
fprintf('Estimated Bias Proxy (Residual-Control Correlation): %.4f\n', estimated_bias);
fprintf('Expected Real-World Error Bound (Ghost Variance): %.6f\n', expected_bias_mse);

% --- Plotting the Identification Error ---
figure('Name', 'Closed-Loop Identification Error', 'Position', [150, 150, 800, 400]);
x_eval = linspace(-0.5, 0.5, 100);
y_ideal = B_true * x_eval.^3;
y_nn = arrayfun(nn_predict, x_eval);

plot(x_eval, y_ideal, 'k--', 'LineWidth', 2, 'DisplayName', 'Ideal Optimal Mapping ($B x^3$)'); hold on; grid on;
plot(x_eval, y_nn, 'r-', 'LineWidth', 2, 'DisplayName', 'Offline NN Prediction');
xlabel('State $x_k$', 'Interpreter', 'latex');
ylabel('Modeled Nonlinearity', 'Interpreter', 'latex');
title('Offline Network Prediction vs. True Dynamics', 'Interpreter', 'latex', 'FontSize', 14);
legend('Location', 'northwest', 'Interpreter', 'latex');


fprintf('\n=== QUANTIFYING THE ACTION ERROR (DEPLOYMENT DRIFT) ===\n');

% 1. Reconstruct the Baseline Control Effort used during Phase 2 (Offline NN Run)
% We need this to calculate both the empirical error and the Jacobians later.
u_base_off = zeros(1, demo_steps);
for k = 1:demo_steps
    u_ff = x_ref_deploy(k+1) - A_nom * x_ref_deploy(k);
    if k==1
        vref = 0; 
        vel2 = 0; 
    else
        vref = x_ref_deploy(k) - x_ref_deploy(k-1);
        vel2 = x_off(k) - x_off(k-1); 
    end
    u_base_off(k) = u_ff - Kp*(x_off(k) - x_ref_deploy(k)) - Kd*(vel2 - vref);
end

% --- EMPIRICAL ACTION ERROR ESTIMATION (REAL-WORLD OBSERVABLE) ---
% A. Training Error (Loss on D_off)
% Using 'inputs' and 'targets' from Phase 2 of the original workspace
nn_train_preds = arrayfun(nn_predict, inputs);
mse_train = mean((nn_train_preds - targets).^2);

% B. Deployment Error (Loss on D_deploy)
nn_deploy_preds = arrayfun(nn_predict, x_off(1:demo_steps));
r_obs_deploy = zeros(1, demo_steps);

for k = 1:demo_steps
    % Reconstruct what the total applied control was
    u_total_applied = u_base_off(k) - nn_deploy_preds(k);
    
    % The observable real-world residual (what the NN should have predicted)
    % Notice this relies ONLY on known nominal dynamics (A_nom) and sensors (x)
    r_obs_deploy(k) = x_off(k+1) - (A_nom * x_off(k) + u_total_applied);
end

mse_deploy = mean((nn_deploy_preds - r_obs_deploy).^2);

% C. The Empirical Action Error (The Generalization Gap)
empirical_action_error = mse_deploy - mse_train;

fprintf('Offline Training MSE (on D_off): %f\n', mse_train);
fprintf('Deployment MSE (on D_deploy): %f\n', mse_deploy);
fprintf('Empirical Action Error (Generalization Gap): %f\n\n', empirical_action_error);
% 2. Define the Plant Black Box (without the noise w_k to isolate deterministic drift)

% Using the parameters (A_nom, B_true) already in the workspace
plant_blackbox = @(x_val, u_val) phys_cap(A_nom * x_val + B_true * x_val^3 + u_val);

% 3. Reconstruct the Baseline Control Effort used during Phase 2 (Offline NN Run)
u_base_off = zeros(1, demo_steps);
for k = 1:demo_steps
    u_ff = x_ref_deploy(k+1) - A_nom * x_ref_deploy(k);
    if k==1
        vref = 0; 
        vel2 = 0; 
    else
        vref = x_ref_deploy(k) - x_ref_deploy(k-1);
        vel2 = x_off(k) - x_off(k-1); 
    end
    u_base_off(k) = u_ff - Kp*(x_off(k) - x_ref_deploy(k)) - Kd*(vel2 - vref);
end

% 4. Compute the Sensitivity Accumulation using the native 'net' object
[S_hist, S_norm] = compute_action_error_sensitivity(...
    plant_blackbox, net, mu_in, sigma_in, mu_t, sigma_t, x_off, u_base_off);

fprintf('Max Norm of State Sensitivity (||S_k||): %.2f\n', max(S_norm));

% --- Plotting the Action Error Divergence ---
figure('Name', 'Action Error: Sensitivity Accumulation', 'Position', [180, 180, 800, 400]);
plot(1:demo_steps, S_norm, 'r-', 'LineWidth', 2); hold on; grid on;
xlabel('Deployment Time Step $k$', 'Interpreter', 'latex');
ylabel('Sensitivity Magnitude $||S_k||_F$', 'Interpreter', 'latex');
title('Accumulation of State Sensitivity (Control Authority)', 'Interpreter', 'latex', 'FontSize', 14);

fprintf('\n=== QUANTIFYING THE TRACKING ERROR (ONLINE ADAPTATION) ===\n');

% 1. Align the reference trajectory to match the evaluated steps
x_ref_eval = x_ref_deploy(1:demo_steps+1);

% 2. Helper function to calculate Mean Squared Error (MSE) against the reference
calc_tracking_mse = @(x_traj) mean((x_traj - x_ref_eval).^2);

% 3. Compute the Tracking Error for all architectures
mse_base = calc_tracking_mse(x_base);
mse_off  = calc_tracking_mse(x_off);
mse_on   = calc_tracking_mse(x_on);
% mse_ab1  = calc_tracking_mse(x_ab1); % No Buffer
mse_ab2  = calc_tracking_mse(x_ab2); % Sluggish LR
mse_ab3  = calc_tracking_mse(x_ab3); % Unstable LR

fprintf('Baseline Controller MSE:      %f (Constant Lag)\n', mse_base);
fprintf('Static Offline NN MSE:        %f (Action Error Offset)\n', mse_off);
fprintf('Optimal Online NN MSE:        %f (Successfully Converged)\n', mse_on);
% fprintf('Ablation 1 (No Buffer) MSE:   %f (Catastrophic Forgetting)\n', mse_ab1);
fprintf('Ablation 2 (Sluggish LR) MSE: %f (Failed to Track Moving Target)\n', mse_ab2);
fprintf('Ablation 3 (Unstable LR) MSE: %f (Violated Time-Scale Separation)\n', mse_ab3);

% --- Plotting the Absolute Tracking Error Over Time ---
figure('Name', 'Tracking Error Dynamics', 'Position', [200, 200, 1000, 500]);

% Calculate absolute errors over time
err_base = abs(x_base - x_ref_eval);
err_off  = abs(x_off - x_ref_eval);
err_on   = abs(x_on - x_ref_eval);
err_ab1  = abs(x_ab1 - x_ref_eval);
err_ab2  = abs(x_ab2 - x_ref_eval);
err_ab3  = abs(x_ab3 - x_ref_eval);

% Create a 1x2 subplot to cleanly separate the standard models from the ablations
subplot(1, 2, 1);
plot(0:demo_steps, err_base, '--k', 'LineWidth', 1.5, 'DisplayName', 'Baseline'); hold on; grid on;
plot(0:demo_steps, err_off, '-r', 'LineWidth', 1.5, 'DisplayName', 'Offline NN (Offset)');
plot(0:demo_steps, err_on, '-b', 'LineWidth', 2.5, 'DisplayName', 'Optimal Online NN');
xlabel('Deployment Time Step $k$', 'Interpreter', 'latex');
ylabel('Absolute Tracking Error $|x_k - x_{ref}|$', 'Interpreter', 'latex');
title('Standard Architectures', 'Interpreter', 'latex');
legend('Location', 'northwest');
ylim([0, 1.0]);

subplot(1, 2, 2);
plot(0:demo_steps, err_on, '-b', 'LineWidth', 2.5, 'DisplayName', 'Optimal Online NN'); hold on; grid on;
% plot(0:demo_steps, err_ab1, '-', 'Color', '#A0522D', 'LineWidth', 1.5, 'DisplayName', 'No Buffer');
plot(0:demo_steps, err_ab2, '-', 'Color', '#228B22', 'LineWidth', 1.5, 'DisplayName', 'Sluggish LR');
plot(0:demo_steps, err_ab3, '-', 'Color', '#800080', 'LineWidth', 1.5, 'DisplayName', 'Unstable LR');
xlabel('Deployment Time Step $k$', 'Interpreter', 'latex');
ylabel('Absolute Tracking Error $|x_k - x_{ref}|$', 'Interpreter', 'latex');
title('Online Ablation Studies', 'Interpreter', 'latex');
legend('Location', 'northwest');
ylim([0, 1.0]);

fprintf('\n=== LATEX TABLE GENERATION ===\n');

% Print the LaTeX table header
fprintf('\\begin{table}[htpb]\n');
fprintf('\\centering\n');
fprintf('\\caption{Summary of Experimental Error Metrics Across Control Architectures}\n');
fprintf('\\label{tab:error_summary}\n');
fprintf('\\begin{tabular}{lccc}\n');
fprintf('\\hline\n');
fprintf('\\textbf{Architecture} & \\textbf{Id. Error (MSE)} & \\textbf{Action Error} & \\textbf{Tracking Error} \\\\\n');
fprintf('\\hline\n');

% Row 1: Baseline PD (Only Tracking Error applies)
fprintf('Baseline PD            & -        & -        & %.6f \\\\\n', mse_base);

% Row 2: Static Offline NN (All errors apply)
fprintf('Static Offline NN      & %.6f & %.6f & %.6f \\\\\n', exact_id_error, empirical_action_error, mse_off);

% Row 3: Optimal Online NN (Solves Id and Action errors, chases Tracking)
fprintf('Optimal Online NN      & -        & -        & %.6f \\\\\n', mse_on);

% % Row 4: Ablation 1 (No Buffer)
% fprintf('Ablation 1 (No Buffer) & -        & -        & %.6f \\\\\n', mse_ab1);

% Row 5: Ablation 2 (Sluggish LR)
fprintf('Ablation 2 (Sluggish LR)& -       & -        & %.6f \\\\\n', mse_ab2);

% Row 6: Ablation 3 (Unstable LR)
fprintf('Ablation 3 (Unstable LR)& -       & -        & %.6f \\\\\n', mse_ab3);

% Print the LaTeX table footer
fprintf('\\hline\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n\n');