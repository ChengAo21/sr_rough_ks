## 📚 Project Overview & Corresponding Publication
This repository contains the source code and results for the research work presented in our published article: "A Drag Model for Rough Surfaces Learned Using Feature Importance-informed Symbolic Regression" (Journal of Fluids Engineering, 2026).
For full details, please refer to our article: [![DOI](https://img.shields.io/badge/DOI-10.1115%2F1.4070838-blue.svg)](https://doi.org/10.1115/1.4070838)

Detailed mathematical principles and formula derivations of the PCE can be found in Section 4.4 and Appendix C of our article.

If you find this repository useful, please consider giving a star ⭐ and cite our paper.
```
@article{cheng2026drag,
  title={A Drag Model for Rough Surfaces Learned Using Feature Importance-informed Symbolic Regression},
  author={Cheng, Ao and Zhou, Zhideng and Yang, Xiaolei and He, Guo-Wei},
  journal={Journal of Fluids Engineering},
  pages={1-44},
  year={2026},
  month={01},
  doi={10.1115/1.4070838},
}
```

## 🌟 Project Structure
```
├── code_files            # Code for Polynomial Chaos Expansion
    ├── pce_core.py         # PolyChaos Class
    ├── pce_utils.py        # Helper Functions
    └── run_analysis.py     # User Model, Parameters, and Execution
└── results
    ├── csvFiles          # Roughness statistics and PySR searching results
    ├── SR_train          # Symbolic training configurations
    ├── subFunc_plot      # Visualization of the sub-functions
    └── loss_plot         # Symbolic model loss visualization
```

## 🚀 Polynomial Chaos Expansion (PCE)
Core steps for applying PCE to build a surrogate model, quantify uncertainty, and perform global sensitivity analysis are summarized below.
```
├── PCE_Implementation       
│   ├── PolyChaos            
│   │   ├── __init__         # Initialize dimension, order, distributions, and parameters
│   │   ├── create_instance  # Generate multi-index set and the multivariate polynomial basis functions
│   │   ├── generate_multi_index # Generate the total-order multi-index set
│   │   ├── norm_factor      # Compute the normalization factor for orthogonal polynomials
│   │   ├── create_latin_hypercube_samples # Generate input samples using Latin Hypercube Sampling (LHS)
│   │   ├── regression_fit   # Fit PCE coefficients using Least-Squares Regression (LSR)
│   │   ├── norm_fit         # Compute statistical moments (mean and variance)
│   │   ├── sobol            # Calculate Sobol Sensitivity Indices
│   │   └── evaluate         # Evaluate the PCE model at given input points

│   └── main                 # Execution Flow
│       ├── Initialization   
│       ├── Sampling         # Generate LHS points
│       ├── Evaluate_Model   # Evaluate Your Model at the LHS points
│       ├── Fit_PCE          # Determine PCE coefficients
│       ├── Compute_Stats    # Compute mu and sigma
│       ├── Validate_PCE     # Validate the PCE on your own data points
│       └── Sensitivity      # Compute Sobol sensitivity indices

```

## 📖 Physical Interpretability of Model
### Visualization of the sub-functions and their pairwise interactions
![schematic](ResponseSurface.png)
