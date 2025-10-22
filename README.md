# 🌟 Polynomial Chaos Expansion (PCE) Framework for Uncertainty Quantification

This repository outlines the core mathematical formulations and algorithmic steps for implementing a **Polynomial Chaos Expansion (PCE)** framework, a powerful technique for uncertainty quantification and sensitivity analysis.

The framework utilizes a multi-dimensional vector of random variables, $\boldsymbol{\xi} = (\xi_1, \xi_2, \dots, \xi_D)$, where $D$ is the number of input variables.

---

## 1. Univariate Orthogonal Polynomials

For each input random variable $\xi_i$, an orthogonal polynomial family $\Phi_k^{(i)}(\xi_i)$ is selected. This family is specifically chosen such that its **weight function** $\omega_i(\xi_i)$ is equivalent to the **probability density function (PDF)** of $\xi_i$.

### Orthogonality Property

The univariate polynomials satisfy the following orthogonality condition:

$$
\mathbb{E}[\Phi_k^{(i)}(\xi_i) \Phi_l^{(i)}(\xi_i)] = \int \Phi_k^{(i)}(\xi_i) \Phi_l^{(i)}(\xi_i) \omega_i(\xi_i) d\xi_i = \delta_{kl} \Vert \Phi_k^{(i)} \Vert^2
$$

where $\delta_{kl}$ is the Kronecker delta (1 if $k=l$, 0 otherwise), and $\Vert \Phi_k^{(i)} \Vert^2$ is the squared norm of the polynomial.

---

## 2. Multivariate PCE Basis Functions

The multi-dimensional PCE basis functions $\Psi_{\boldsymbol{\alpha}}(\boldsymbol{\xi})$ are constructed through the **tensor product** of the univariate orthogonal polynomials:

$$
\Psi_{\boldsymbol{\alpha}}(\boldsymbol{\xi}) = \Psi_{(\alpha_{1},\alpha_{2},\cdots,\alpha_{D})}(\xi_1, \xi_2, \dots, \xi_D) = \prod_{s=1}^{D} \Phi_{\alpha_{s}}^{(s)}(\xi_s)
$$

Here, $\boldsymbol{\alpha} = (\alpha_{1},\alpha_{2},\cdots,\alpha_{D})$ is the **multi-index**, where $\alpha_s$ denotes the order of the polynomial for variable $\xi_s$.

### Multivariate Orthogonality

Given the joint weight function $\boldsymbol{\omega}(\boldsymbol{\xi}) = \prod_{i=1}^{D} \omega_i(\xi_i)$, the expected value for the product of any two multivariate basis functions, $\Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi})$ and $\Psi_{\boldsymbol{\alpha_l}}(\boldsymbol{\xi})$, is:

$$
\begin{aligned}
  \mathbb{E}[\Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi}) \Psi_{\boldsymbol{\alpha_l}}(\boldsymbol{\xi})] &= \int_{\Omega} \left( \prod_{i=1}^{D} \Phi_{\alpha_i^k}^{(i)}(\xi_i) \right) \left( \prod_{i=1}^{D} \Phi_{\alpha_i^l}^{(i)}(\xi_i) \right) \left( \prod_{i=1}^{D} \omega_i(\xi_i) \right) d\boldsymbol{\xi}\\
   &= \prod_{i=1}^{D} \left( \int_{\Omega_i} \Phi_{\alpha_i^k}^{(i)}(\xi_i) \Phi_{\alpha_i^l}^{(i)}(\xi_i) \omega_i(\xi_i) d\xi_i \right)
\end{aligned}
$$

Due to the univariate orthogonality (Section 1), if $\boldsymbol{\alpha_k} \neq \boldsymbol{\alpha_l}$ (meaning at least one dimension $i^*$ has $\alpha_{i^*}^k \neq \alpha^l_{i^*}$), the entire product is **zero**. If $\boldsymbol{\alpha_k} = \boldsymbol{\alpha_l}$, the result is $\prod_{i=1}^{D} \Vert \Phi_{\alpha_i^k}^{(i)} \Vert^2$.

---

## 3. Core PCE Algorithm

The PCE surrogate model for an output quantity of interest $Q(\boldsymbol{\xi})$ is expressed as a series expansion:

$$
Q(\boldsymbol{\xi}) \approx \sum_{k=0}^{N_t-1} c_k \Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi})
$$

### A. Standardization and Basis Count

1.  **Standardization:** Input random variables $\boldsymbol{\xi}$ must be standardized (e.g., mapped to $\mathcal{U}[-1, 1]$ or $\mathcal{N}(0, 1)$) to correspond to standard orthogonal polynomial types (e.g., Legendre or Hermite).
2.  **Basis Term Count ($N_t$):** The total number of basis terms is determined by the maximum prescribed polynomial order $P$ and the number of input variables $D$:
    $$
    N_t=\binom{D+P}{P}
    $$

### B. Coefficient Calculation (Least Squares)

The polynomial coefficients $c_k$ are determined by solving an overspecified system of algebraic equations using the least squares method.

1.  **Sampling:** Generate $N_s$ Latin Hypercube Sampling (LHS) points, $\boldsymbol{\xi}_j$, in the standardized random variable space, where $N_s > N_t$.
2.  **Model Evaluation:** Compute the model output $Q(\boldsymbol{\xi}_j)$ for each sample point.
3.  **Linear System:** The coefficients $c_k$ are solved from the system $A \cdot \mathbf{c} = \mathbf{Q}$:

$$
\begin{pmatrix}
Q(\boldsymbol{\xi}_1) \\
Q(\boldsymbol{\xi}_2) \\
\vdots \\
Q(\boldsymbol{\xi}_{N_s})
\end{pmatrix}
=
\begin{pmatrix}
\Psi_{\boldsymbol{\alpha}_0}(\boldsymbol{\xi}_1) & \Psi_{\boldsymbol{\alpha}_1}(\boldsymbol{\xi}_1) & \cdots & \Psi_{\boldsymbol{\alpha}_{N_t-1}}(\boldsymbol{\xi}_1) \\
\Psi_{\boldsymbol{\alpha}_0}(\boldsymbol{\xi}_2) & \Psi_{\boldsymbol{\alpha}_1}(\boldsymbol{\xi}_2) & \cdots & \Psi_{\boldsymbol{\alpha}_{N_t-1}}(\boldsymbol{\xi}_2) \\
\vdots & \vdots & \ddots & \vdots \\
\Psi_{\boldsymbol{\alpha}_0}(\boldsymbol{\xi}_{N_s}) & \Psi_{\boldsymbol{\alpha}_1}(\boldsymbol{\xi}_{N_s}) & \cdots & \Psi_{\boldsymbol{\alpha}_{N_t-1}}(\boldsymbol{\xi}_{N_s})
\end{pmatrix}
\begin{pmatrix}
c_0 \\
c_1 \\
\vdots \\
c_{N_t-1}
\end{pmatrix}
$$

---

## 4. Moments and Sensitivity Analysis

The orthogonality property of the PCE basis enables simple analytical calculation of statistical moments and global sensitivity indices (Sobol indices).

### A. Mean Value $\mathbb{E}[Q(\boldsymbol{\xi})]$

The basis function $\Psi_{\boldsymbol{\alpha_0}}(\boldsymbol{\xi})$ is the constant $\mathbf{1}$ (zero-order for all variables). Due to orthogonality, $\mathbb{E}[\Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi})] = 0$ for $k \geq 1$.

$$
\begin{aligned}
  \mathbb{E}[Q(\boldsymbol{\xi})] &= \mathbb{E}[\sum_{k=0}^{N_t-1} c_k \Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi})] \\
  &= c_0 + \sum_{k=1}^{N_t-1} c_k \cdot 0 \\
  &= \mathbf{c_0}
\end{aligned}
$$

### B. Variance $\mathrm{Var}[Q(\boldsymbol{\xi})]$

Using the orthogonality of the basis, the variance simplifies significantly:

$$
\begin{aligned}
  \mathrm{Var}[Q(\boldsymbol{\xi})] &= \mathbb{E}[Q(\boldsymbol{\xi})^2] - (\mathbb{E}[Q(\boldsymbol{\xi})])^2 \\
  &= \sum_{k=1}^{N_t-1} c_k^2 \mathbb{E}[\Psi_{\boldsymbol{\alpha_k}}^2(\boldsymbol{\xi})] \\
  &= \sum_{k=1}^{N_t-1} c_k^2 \Vert \Psi_{\boldsymbol{\alpha_k}} \Vert^2
\end{aligned}
$$

**Note:** $\mathbb{E}[\Psi_{\boldsymbol{\alpha_k}}^2(\boldsymbol{\xi})]$ is the squared norm $\Vert \Psi_{\boldsymbol{\alpha_k}} \Vert^2$, which is calculated as $\prod_{i=1}^{D} \Vert \Phi_{\alpha_i^k}^{(i)} \Vert^2$. If the basis is orthonormal, $\Vert \Psi_{\boldsymbol{\alpha_k}} \Vert^2 = 1$.

### C. Sobol Indices ($S_I$)

Sobol indices are analytically determined by grouping coefficients $c_k$ corresponding to a specific subset of input features $I \subseteq \{1, 2, \dots, D\}$.

For any subset of input features $I$, the Sobol index $S_I$ is calculated as the ratio of the partial variance associated with $I$ to the total variance. The partial variance is obtained by summing the squared coefficients and their norms for all basis functions $\Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi})$ whose multi-index $\boldsymbol{\alpha_k}$ is non-zero *only* for the dimensions in set $I$.

**Example: Second-Order Joint Sobol Index $S_{1,2}$**

$S_{1,2}$ is calculated by considering all basis functions $\Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi})$ for which the multi-index satisfies:
$$
\boldsymbol{\alpha_k}=(\alpha_1^k, \alpha_2^k, 0, \dots, 0)
$$
where $\alpha_1^k \geq 1$ and $\alpha_2^k \geq 1$.
