% run_monte_carlo.m
% Runs a 100-iteration Monte Carlo simulation of the Benchmark_exo environment.
% Filters out unstable runs for the Static Offline NN and outputs mean/variance.

clc;
num_runs = 100;

fprintf('Preparing Monte Carlo simulation...\n');

% 1. Create a "silent" version of Benchmark_exo.m
fid = fopen('Benchmark_exo.m', 'rt');

% Add a safety check in case the file is missing
if fid == -1
    error('Could not open Benchmark_exo.m! Please check that the file is in your Current Folder: %s', pwd);
end

code = fread(fid, '*char')';
fclose(fid);

% Strip workspace clearing commands
code = strrep(code, 'clear;', '%clear;');
code = strrep(code, 'clc;', '%clc;');
code = strrep(code, 'close all;', '%close all;');

% Strip plot generation commands
code = regexprep(code, 'figure\(.*?\);', '%figure;');
code = regexprep(code, 'subplot\(.*?\);', '%subplot;');
code = regexprep(code, 'plot\(.*?\);', '%plot;');
code = regexprep(code, 'xlabel\(.*?\);', '%xlabel;');
code = regexprep(code, 'ylabel\(.*?\);', '%ylabel;');
code = regexprep(code, 'title\(.*?\);', '%title;');
code = regexprep(code, 'legend\(.*?\);', '%legend;');
code = regexprep(code, 'xlim\(.*?\);', '%xlim;');
code = regexprep(code, 'ylim\(.*?\);', '%ylim;');
code = regexprep(code, 'xline\(.*?\);', '%xline;');
code = regexprep(code, 'yline\(.*?\);', '%yline;');

fid = fopen('Benchmark_exo_silent.m', 'wt');
fwrite(fid, code);
fclose(fid);

% Preallocate data storage arrays
mse_base_all   = [];
mse_off_all    = [];
mse_on_all     = [];
mse_ab2_all    = [];
mse_ab3_all    = [];
exact_id_all   = [];
emp_action_all = [];

diverged_count  = 0;
converged_count = 0;

fprintf('Running %d iterations (this will take a few minutes)...\n', num_runs);

for i = 1:num_runs
    if mod(i, 10) == 0
        fprintf('Iteration %d / %d...\n', i, num_runs);
    end
    
    % Run the silent script. evalc suppresses command window spam from trainlm.
    evalc('Benchmark_exo_silent');
    
    % 2. Calculate the Tracking Error specifically for the Static Offline NN
    x_ref_eval = x_ref_deploy(1:demo_steps+1);
    calc_tracking_mse = @(x_traj) mean((x_traj - x_ref_eval).^2);
    
    current_mse_off = calc_tracking_mse(x_off);
    
    % 3. Check for Divergence
    % If the tracking MSE is exceptionally large, the system has lost stability.
    if current_mse_off > 1.0 
        diverged_count = diverged_count + 1;
        continue; % Skip adding this run to our stable statistics
    end
    
    converged_count = converged_count + 1;
    
    % 4. Extract metrics for stable runs
    mse_base_all(end+1) = calc_tracking_mse(x_base);
    mse_off_all(end+1)  = current_mse_off;
    mse_on_all(end+1)   = calc_tracking_mse(x_on);
    mse_ab2_all(end+1)  = calc_tracking_mse(x_ab2);
    mse_ab3_all(end+1)  = calc_tracking_mse(x_ab3);
    
    % --- Identification Error Calculation ---
    x_calib = x(1:N_steps); 
    u_base_calib = u_total_history(1:N_steps);
    [exact_id_error, ~, ~] = compute_identification_error(nn_predict, x_calib, u_base_calib, x_ref_calib, B_true);
    exact_id_all(end+1) = exact_id_error;
    
    % --- Action Error (Generalization Gap) Calculation ---
    nn_train_preds = arrayfun(nn_predict, inputs);
    mse_train = mean((nn_train_preds - targets).^2);
    
    nn_deploy_preds = arrayfun(nn_predict, x_off(1:demo_steps));
    r_obs_deploy = zeros(1, demo_steps);
    u_base_off = zeros(1, demo_steps);
    
    for k = 1:demo_steps
        u_ff = x_ref_deploy(k+1) - A_nom * x_ref_deploy(k);
        if k==1
            vref = 0; vel2 = 0; 
        else
            vref = x_ref_deploy(k) - x_ref_deploy(k-1);
            vel2 = x_off(k) - x_off(k-1); 
        end
        u_base_off(k) = u_ff - Kp*(x_off(k) - x_ref_deploy(k)) - Kd*(vel2 - vref);
        
        u_total_applied = u_base_off(k) - nn_deploy_preds(k);
        r_obs_deploy(k) = x_off(k+1) - (A_nom * x_off(k) + u_total_applied);
    end
    mse_deploy = mean((nn_deploy_preds - r_obs_deploy).^2);
    emp_action_all(end+1) = mse_deploy - mse_train;
end

% Clean up the temporary file
delete('Benchmark_exo_silent.m');

% 5. Print Results & Generate LaTeX Table
fprintf('\n=== MONTE CARLO RESULTS (%d RUNS) ===\n', num_runs);
fprintf('Stable (Converged) Runs: %d\n', converged_count);
fprintf('Unstable (Diverged) Runs: %d\n\n', diverged_count);

% Helper to format strings as "Mean (\sigma^2: Variance)"
fmt = @(data) sprintf('%.4f (\\\\sigma^2: %.4f)', mean(data), var(data));

fprintf('=== LATEX TABLE GENERATION ===\n');
fprintf('\\begin{table}[htpb]\n');
fprintf('\\centering\n');
fprintf('\\caption{Monte Carlo Summary of Experimental Error Metrics (%d Stable Runs, %d Diverged)}\n', converged_count, diverged_count);
fprintf('\\label{tab:mc_error_summary}\n');
fprintf('\\resizebox{\\columnwidth}{!}{%%\n');
fprintf('\\begin{tabular}{lccc}\n');
fprintf('\\hline\n');
fprintf('\\textbf{Architecture} & \\textbf{Id. Error (MSE)} & \\textbf{Action Error} & \\textbf{Tracking Error} \\\\\n');
fprintf('\\hline\n');

fprintf('Baseline PD            & - & - & %s \\\\\n', fmt(mse_base_all));
fprintf('Static Offline NN      & %s & %s & %s \\\\\n', fmt(exact_id_all), fmt(emp_action_all), fmt(mse_off_all));
fprintf('Optimal Online NN      & - & - & %s \\\\\n', fmt(mse_on_all));
fprintf('Ablation 2 (Sluggish)  & - & - & %s \\\\\n', fmt(mse_ab2_all));
fprintf('Ablation 3 (Unstable)  & - & - & %s \\\\\n', fmt(mse_ab3_all));

fprintf('\\hline\n');
fprintf('\\end{tabular}%%\n');
fprintf('}\n');
fprintf('\\end{table}\n');