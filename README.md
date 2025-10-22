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
│   │   ├── evaluate         # Evaluates the PCE model at given input points
│   │   └── __str__          # Provides a summary printout of the PCE instance
│   ├── PceSmolyakGrid       # Smolyak Sparse Grid Class (Implementation for Spectral Projection, though LSR is used in the main function)
│   │   ├── __init__         # Initializes the Smolyak grid
│   │   ├── smolyak_sparse_grid # Generates sparse grid points, weights, and unique point indexing
│   │   ├── index_with_sum   # Helper: Finds multi-indices that sum to a specific value
│   │   └── growth_rule      # Helper: Determines the number of integration points based on the level and rule
│   ├── Quadrature_Routines  # Auxiliary functions for numerical integration (Quadrature)
│   │   ├── columnize        # Reshapes and concatenates arrays into columns
│   │   ├── clencurt         # Computes Clenshaw-Curtis quadrature nodes and weights
│   │   └── quad_coeff       # Unified interface for Gaussian (Gauss-Legendre, Gauss-Hermite) and Clenshaw-Curtis coefficients
│   └── main                 # Example of PCE Usage (The Execution Flow)
│       ├── Initialization   # Instantiates the PolyChaos model
│       ├── Sampling         # Calls LHS to generate samples
│       ├── Model_Evaluation # Evaluates Your Model at the sample points
│       ├── Fitting          # Calls regression_fit to determine PCE coefficients
│       ├── Statistics       # Calls norm_fit to compute $\mu$ and $\sigma$
│       ├── Validation       # Calls compare_on_myData for model comparison/validation
│       └── Sensitivity      # Calls sobol to compute and print the sensitivity indices
```
