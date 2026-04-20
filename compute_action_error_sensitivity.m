function [S_history, S_norm] = compute_action_error_sensitivity(plant_bb, net_obj, mu_in, sigma_in, mu_t, sigma_t, x_traj, u_base_traj)
    % COMPUTE_ACTION_ERROR_SENSITIVITY Numerically calculates state sensitivity (S_k)
    % using central finite differences directly on the MATLAB net object.
    
    eps_x = 1e-5;
    eps_u = 1e-5;
    eps_t = 1e-5;
    
    N = length(x_traj) - 1;
    
    % Extract the flat parameter vector directly from the network object
    theta_flat = getx(net_obj);
    num_params = length(theta_flat);
    
    S_k = zeros(1, num_params); 
    S_history = zeros(N, num_params);
    S_norm = zeros(1, N);
    
    % Helper function for normalized NN prediction
    predict_nn = @(net_in, x_val) sigma_t .* net_in((x_val - mu_in)./sigma_in) + mu_t;
    
    for k = 1:N
        x_k = x_traj(k);
        u_base = u_base_traj(k);
        
        % 1. Evaluate nominal control effort
        u_nn_nom = predict_nn(net_obj, x_k);
        u_tot_nom = u_base - u_nn_nom; 
        
        % 2. Plant Jacobians
        df_dx = (plant_bb(x_k + eps_x, u_tot_nom) - plant_bb(x_k - eps_x, u_tot_nom)) / (2 * eps_x);
        df_du = (plant_bb(x_k, u_tot_nom + eps_u) - plant_bb(x_k, u_tot_nom - eps_u)) / (2 * eps_u);
        
        % 3. Controller Jacobian: dg/dx
        dg_dx = (predict_nn(net_obj, x_k + eps_x) - predict_nn(net_obj, x_k - eps_x)) / (2 * eps_x);
        
        % 4. Parameter Jacobian: dg/dtheta (Control Authority)
        dg_dtheta = zeros(1, num_params);
        for p = 1:num_params
            % Perturb parameter positively
            theta_plus = theta_flat; theta_plus(p) = theta_plus(p) + eps_t;
            net_plus = setx(net_obj, theta_plus);
            g_t_plus = predict_nn(net_plus, x_k);
            
            % Perturb parameter negatively
            theta_minus = theta_flat; theta_minus(p) = theta_minus(p) - eps_t;
            net_minus = setx(net_obj, theta_minus);
            g_t_minus = predict_nn(net_minus, x_k);
            
            dg_dtheta(1, p) = (g_t_plus - g_t_minus) / (2 * eps_t);
        end
        
        % 5. Assemble Closed-Loop Jacobians
        Phi_k = df_dx - df_du * dg_dx; 
        Lambda_k = -df_du * dg_dtheta;
        
        % 6. Propagate State Sensitivity
        S_k = Phi_k * S_k + Lambda_k;
        
        S_history(k, :) = S_k;
        S_norm(k) = norm(S_k); 
    end
end