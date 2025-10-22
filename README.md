# 🌟 Project Structure
```
├── code_files            # Code for Polynomial Chaos Expansion
└── results
    ├── csvFiles          # Roughness statistics and PySR searching results
    └── loss_plot         # Symbolic model loss visualization
```

# 🚀 Polynomial Chaos Expansion (PCE)
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
