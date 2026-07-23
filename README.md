# Action-Error: Training and Deploying Neural Networks in the Control Loop

This repository contains the MATLAB source code and simulation environment for the paper: **"Learning Changes the World and Then What Was Learned has Changed: Training and Deploying Neural Networks in the Control Loop,"** submitted to ICINCO 2026[cite: 3, 4]. 

## 📖 Background & Justification

Neural networks (NNs) offer critical adaptability for controlling complex nonlinear systems, such as legged robots and exoskeletons, where classical modeling is often intractable[cite: 3]. However, deploying offline-trained neural networks into physical closed-loop control systems consistently degrades their offline-validated performance[cite: 3]. 

This repository empirically validates the mathematical taxonomy of sequential failure modes that cause this degradation:
1. **The Closed-Loop Identification Error:** During offline data collection, baseline controllers must remain active to stabilize the plant. This enforces a statistical cross-correlation between the control inputs and unmeasured environmental noise, causing the NN to implicitly overfit to process noise rather than the true deterministic dynamics[cite: 3].
2. **The Action Error:** When the static, offline-trained NN is deployed, it fundamentally alters the system dynamics it originally mapped[cite: 3]. Because the network possesses "Control Authority," its parametric sub-optimality mathematically guarantees a trajectory divergence, pushing the system into out-of-distribution states[cite: 3].
3. **The Tracking Error:** Recovering from the Action Error requires continuous online learning. However, because the optimal parameter is a moving target, the system must continuously minimize the Tracking Error through bounded, time-scale separated adaptation to avoid destabilizing the plant[cite: 3].

We validate these concepts using a 1-DOF nonlinear Duffing oscillator, which serves as a trackable model for wearable robotics (e.g., an exoskeleton joint exhibiting unmodeled cubic stiffness)[cite: 3].

## 🗂️ Repository Structure

* **`Benchmark_exo.m`**: The primary execution script[cite: 7]. It simulates the 1-DOF Duffing oscillator across three phases:
  * *Phase 1:* Calibration via a baseline PD controller to collect small-amplitude closed-loop data[cite: 7].
  * *Phase 2:* Offline NN training using the Levenberg-Marquardt algorithm[cite: 7].
  * *Phase 3:* Parallel evaluation of standard architectures (Baseline, Static Offline NN, Optimal Online NN) and ablation studies (Sluggish Learning Rate, Unstable Learning Rate) on a large-amplitude trajectory[cite: 7].
* **`analyze_errors.m`**: An analysis script meant to be run immediately after `Benchmark_exo.m`[cite: 6]. It calculates and visualizes the Exact Identification Error, the proxy bias, the Action Error (generalization gap), and the continuous Tracking Error[cite: 6].
* **`run_monte_carlo.m`**: Executes a 100-iteration Monte Carlo simulation of the environment to evaluate stochastic noise realizations[cite: 5]. It filters out divergent runs (where unmitigated Action Error causes instability) and generates a LaTeX-formatted table of the aggregated tracking metrics[cite: 5].
* **`compute_action_error_sensitivity.m`**: A helper function that numerically calculates the state sensitivity matrix ($S_k$) using central finite differences directly on the MATLAB network object[cite: 8].
* **`compute_identification_error.m`**: A helper function that quantifies the offline identification bias, computing the Exact Functional Error, an Instrumental Variable proxy ratio, and the "Ghost Variance"[cite: 9].

## 🚀 Usage Instructions

**Prerequisites:** 
* MATLAB 
* Deep Learning Toolbox (required for `trainlm` and feedforward network initialization)[cite: 3, 7].

**To run a single simulation and view the dynamics:**
1. Open MATLAB and navigate to the repository directory.
2. Run `Benchmark_exo.m`[cite: 7]. This will train the networks and output a 2x2 grid plotting the time series and return maps for the standard architectures and ablation studies[cite: 7].
3. Run `analyze_errors.m` to calculate the exact theoretical error metrics and generate the corresponding sensitivity and tracking error charts[cite: 6].

**To generate the statistical data (Table 1 in the paper):**
1. Run `run_monte_carlo.m`[cite: 5]. This script suppresses standard plotting output and will take a few minutes to execute 100 iterations[cite: 5]. It will print the final aggregated data directly to the command window[cite: 5].

## 📄 License
This project is licensed under the GNU General Public License v3.0 - see the `LICENSE` file for details[cite: 10].