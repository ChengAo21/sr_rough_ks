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
├── PCE_Implementation       # Core Code for Polynomial Chaos Expansion
│   ├── PolyChaos            # Polynomial Chaos Expansion Class (The main component)
│   │   ├── __init__         # Initializes PCE: sets dimension, order, distributions, and parameters
│   │   ├── create_instance  # Generates multi-index set and the multivariate polynomial basis functions
│   │   ├── generate_multi_index # Generates the total-order multi-index set
│   │   ├── norm_factor      # Computes the normalization factor for orthogonal polynomials
│   │   ├── create_latin_hypercube_samples # Generates input samples using Latin Hypercube Sampling (LHS)
│   │   ├── regression_fit   # Fits PCE coefficients using Least-Squares Regression (LSR)
│   │   ├── norm_fit         # Computes the statistical moments (mean, $\mu$, and variance, $\sigma^2$)
│   │   ├── sobol            # Calculates Sobol Sensitivity Indices ($S_i$, $S_{ij}$, etc.)
│   │   └── evaluate         # Evaluates the PCE model at given input points

│   └── main                 # Example of PCE Usage (The Execution Flow)
│       ├── Initialization   # Instantiates the PolyChaos model
│       ├── Sampling         # Calls LHS to generate samples
│       ├── Model_Evaluation # Evaluates Your Model at the sample points
│       ├── Fitting          # Calls regression_fit to determine PCE coefficients
│       ├── Statistics       # Calls norm_fit to compute $\mu$ and $\sigma$
│       ├── Validation       # Calls compare_on_myData for model comparison/validation
│       └── Sensitivity      # Calls sobol to compute and print the sensitivity indices
```
