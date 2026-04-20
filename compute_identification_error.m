function [exact_error, est_bias_proxy, expected_bias_mse] = compute_identification_error(nn_predict, x_data, u_base_data, x_ref_data, B_true)
    % COMPUTE_IDENTIFICATION_ERROR Quantifies the offline identification bias.
    
    N = length(x_data);
    nn_outputs = zeros(1, N);
    ideal_outputs = zeros(1, N);
    
    for i = 1:N
        nn_outputs(i) = nn_predict(x_data(i));
        % The exact unmodeled dynamics the NN is trying to learn
        ideal_outputs(i) = B_true * x_data(i)^3; 
    end
    
    % 1. Exact Functional Error (Known Optimal)
    exact_error = mean((nn_outputs - ideal_outputs).^2);
    
    % 2. Estimated Bias Proxy (Instrumental Variable Approach)
    nn_outputs_col = nn_outputs(:);
    u_base_col = u_base_data(:);
    
    % Correlation with the noisy baseline control
    corr_with_noise_source = abs(corr(nn_outputs_col, u_base_col));
    
    % Correlation with the clean, deterministic instrument
    x_ref_aligned = x_ref_data(1:N); 
    x_ref_col = x_ref_aligned(:);
    corr_with_clean_instrument = abs(corr(nn_outputs_col, x_ref_col));
    
    % The Proxy Ratio
    est_bias_proxy = corr_with_noise_source / (corr_with_clean_instrument + 1e-6);
    
    % 3. Expected Physical Bias (The "Ghost Variance")
    nn_variance = var(nn_outputs_col);
    r_squared_noise = corr_with_noise_source^2;
    expected_bias_mse = r_squared_noise * nn_variance;
end