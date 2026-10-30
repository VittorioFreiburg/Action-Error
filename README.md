# Action-Error: Training and Deploying Neural Networks in the Control Loop

[![Paper DOI: 10.5220/0015198100004979](https://img.shields.io/badge/Paper%20DOI-10.5220%2F0015198100004979-blue.svg)](https://doi.org/10.5220/0015198100004979)
[![DOI](https://zenodo.org/badge/1215871336.svg)](https://doi.org/10.5281/zenodo.22774300)
[![arXiv](https://img.shields.io/badge/arXiv-2610.22084-b31b1b.svg)](https://arxiv.org/abs/2610.22084)

This repository contains the MATLAB source code and simulation environment for the paper: **"Learning Changes the World and Then What Was Learned Has Changed: Training and Deploying Neural Networks in the Control Loop,"** published in *Proceedings of the 23rd International Conference on Informatics in Control, Automation and Robotics (ICINCO 2026)*, Vol. 1, pages 15–26, ISBN: 978-989-758-870-9, ISSN: 2184-2809, SciTePress.

* **Official Paper DOI:** [10.5220/0015198100004979](https://doi.org/10.5220/0015198100004979)
* **Preprint:** [arXiv:2610.22084 [stat.ML]](https://arxiv.org/abs/2610.22084)
* **Software Archive:** [DOI: 10.5281/zenodo.22774300](https://doi.org/10.5281/zenodo.22774300)

## 📖 Background & Justification

Neural networks (NNs) offer critical adaptability for controlling complex nonlinear systems, such as legged robots and exoskeletons, where classical modeling is often intractable. However, deploying offline-trained neural networks into physical closed-loop control systems consistently degrades their offline-validated performance. 

This repository empirically validates the mathematical taxonomy of sequential failure modes that cause this degradation:
1. **The Closed-Loop Identification Error:** During offline data collection, baseline controllers must remain active to stabilize the plant. This enforces a statistical cross-correlation between the control inputs and unmeasured environmental noise, causing the NN to implicitly overfit to process noise rather than the true deterministic dynamics.
2. **The Action Error:** When the static, offline-trained NN is deployed, it fundamentally alters the system dynamics it originally mapped. Because the network possesses "Control Authority," its parametric sub-optimality mathematically guarantees a trajectory divergence, pushing the system into out-of-distribution states.
3. **The Tracking Error:** Recovering from the Action Error requires continuous online learning. However, because the optimal parameter is a moving target, the system must continuously minimize the Tracking Error through bounded, time-scale separated adaptation to avoid destabilizing the plant.

We validate these concepts using a 1-DOF nonlinear Duffing oscillator, which serves as a trackable model for wearable robotics (e.g., an exoskeleton joint exhibiting unmodeled cubic stiffness).

## 🗂️ Repository Structure

* **`Benchmark_exo.m`**: The primary execution script. It simulates the 1-DOF Duffing oscillator across three phases:
  * *Phase 1:* Calibration via a baseline PD controller to collect small-amplitude closed-loop data.
  * *Phase 2:* Offline NN training using the Levenberg-Marquardt algorithm.
  * *Phase 3:* Parallel evaluation of standard architectures (Baseline, Static Offline NN, Optimal Online NN) and ablation studies (Sluggish Learning Rate, Unstable Learning Rate) on a large-amplitude trajectory.
* **`analyze_errors.m`**: An analysis script meant to be run immediately after `Benchmark_exo.m`. It calculates and visualizes the Exact Identification Error, the proxy bias, the Action Error (generalization gap), and the continuous Tracking Error.
* **`run_monte_carlo.m`**: Executes a 100-iteration Monte Carlo simulation of the environment to evaluate stochastic noise realizations. It filters out divergent runs (where unmitigated Action Error causes instability) and generates a LaTeX-formatted table of the aggregated tracking metrics.
* **`compute_action_error_sensitivity.m`**: A helper function that numerically calculates the state sensitivity matrix ($S_k$) using central finite differences directly on the MATLAB network object.
* **`compute_identification_error.m`**: A helper function that quantifies the offline identification bias, computing the Exact Functional Error, an Instrumental Variable proxy ratio, and the "Ghost Variance".

## 🚀 Usage Instructions

**Prerequisites:** 
* MATLAB 
* Deep Learning Toolbox (required for `trainlm` and feedforward network initialization).

**To run a single simulation and view the dynamics:**
1. Open MATLAB and navigate to the repository directory.
2. Run `Benchmark_exo.m`. This will train the networks and output a 2x2 grid plotting the time series and return maps for the standard architectures and ablation studies.
3. Run `analyze_errors.m` to calculate the exact theoretical error metrics and generate the corresponding sensitivity and tracking error charts.

**To generate the statistical data (Table 1 in the paper):**
1. Run `run_monte_carlo.m`. This script suppresses standard plotting output and will take a few minutes to execute 100 iterations. It will print the final aggregated data directly to the command window.

## 📖 Citation

If you use this codebase or the theoretical taxonomy in your research, please cite our official conference paper and software release:

### Official Conference Paper
> Lippi, V. (2026). Learning Changes the World and Then What Was Learned Has Changed: Training and Deploying Neural Networks in the Control Loop. In *Proceedings of the 23rd International Conference on Informatics in Control, Automation and Robotics - Volume 1: ICINCO*; ISBN 978-989-758-870-9; ISSN 2184-2809, SciTePress, pages 15-26. DOI: [10.5220/0015198100004979](https://doi.org/10.5220/0015198100004979)

```bibtex
@inproceedings{lippi2026learning,
  title     = {Learning Changes the World and Then What Was Learned Has Changed: Training and Deploying Neural Networks in the Control Loop},
  author    = {Lippi, Vittorio},
  booktitle = {Proceedings of the 23rd International Conference on Informatics in Control, Automation and Robotics - Volume 1: ICINCO},
  volume    = {1},
  pages     = {15--26},
  year      = {2026},
  publisher = {SciTePress},
  isbn      = {978-989-758-870-9},
  issn      = {2184-2809},
  doi       = {10.5220/0015198100004979},
  url       = {https://doi.org/10.5220/0015198100004979},
  eprint    = {2610.22084},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}

@software{lippi_sto_code_2026,
  author       = {Lippi, Vittorio},
  title        = {Learning Changes the World and Then What Was Learned Has Changed: Training and Deploying Neural Networks in the Control Loop (Reproduction Suite)},
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.22774300},
  url          = {https://doi.org/10.5281/zenodo.22774300}
}
```

## 📄 License
This project is licensed under the GNU General Public License v3.0 - see the `LICENSE` file for details.
