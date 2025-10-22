# 🚀 Polynomial Chaos Expansion (PCE) for UQ and GSA

This repository provides the core mathematical foundation and implementation steps for using **Polynomial Chaos Expansion (PCE)** as an efficient **surrogate modeling** technique. PCE is primarily used for **Uncertainty Quantification (UQ)** and **Global Sensitivity Analysis (GSA)**, particularly the analytical calculation of Sobol indices.

The framework models an output quantity of interest, $Q(\boldsymbol{\xi})$, as a spectral expansion based on a multi-dimensional vector of standardized random variables $\boldsymbol{\xi} = (\xi_1, \xi_2, \dots, \xi_D)$, where $D$ is the number of inputs.

---

## 1. Basis Function Construction and Orthogonality

PCE relies on the construction of an orthonormal basis tailored to the probability distributions of the input variables.

### Univariate Basis $\Phi_k^{(i)}(\xi_i)$

For each random variable $\xi_i$ with probability density function (PDF) $\omega_i(\xi_i)$, an orthogonal polynomial family $\Phi_k^{(i)}(\xi_i)$ is selected such that $\omega_i(\xi_i)$ serves as its **weight function**.

**Orthogonality Condition:**
$$
\mathbb{E}[\Phi_k^{(i)}(\xi_i) \Phi_l^{(i)}(\xi_i)] = \int \Phi_k^{(i)}(\xi_i) \Phi_l^{(i)}(\xi_i) \omega_i(\xi_i) d\xi_i = \delta_{kl} \Vert \Phi_k^{(i)} \Vert^2
$$

### Multivariate PCE Basis $\Psi_{\boldsymbol{\alpha}}(\boldsymbol{\xi})$

The multivariate basis functions are constructed via a **tensor product** of the univariate polynomials, indexed by the **multi-index** $\boldsymbol{\alpha} = (\alpha_{1},\alpha_{2},\cdots,\alpha_{D})$:

$$
\Psi_{\boldsymbol{\alpha}}(\boldsymbol{\xi}) = \prod_{s=1}^{D} \Phi_{\alpha_{s}}^{(s)}(\xi_s)
$$

**Multivariate Orthogonality:**
Given the joint PDF $\boldsymbol{\omega}(\boldsymbol{\xi}) = \prod_{i=1}^{D} \omega_i(\xi_i)$, the orthogonality extends to the multivariate case:

$$
\mathbb{E}[\Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi}) \Psi_{\boldsymbol{\alpha_l}}(\boldsymbol{\xi})] = \delta_{\boldsymbol{\alpha_k} \boldsymbol{\alpha_l}} \Vert \Psi_{\boldsymbol{\alpha_k}} \Vert^2 = \delta_{\boldsymbol{\alpha_k} \boldsymbol{\alpha_l}} \prod_{i=1}^{D} \Vert \Phi_{\alpha_i^k}^{(i)} \Vert^2
$$
where $\delta_{\boldsymbol{\alpha_k} \boldsymbol{\alpha_l}}$ is the Kronecker delta for the multi-index (1 if $\boldsymbol{\alpha_k} = \boldsymbol{\alpha_l}$, 0 otherwise).

---

## 2. PCE Model Formulation

The PCE model approximates the output $Q(\boldsymbol{\xi})$ as a finite series:

$$
Q(\boldsymbol{\xi}) \approx \sum_{k=0}^{N_t-1} c_k \Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi})
$$

### Coefficient Calculation (Least Squares)

The polynomial coefficients $c_k$ are determined by solving an overspecified system derived from a set of **$N_s$ sample points** ($\boldsymbol{\xi}_j$) and corresponding model outputs ($Q(\boldsymbol{\xi}_j)$).

1.  **Basis Term Count ($N_t$):** The number of terms is typically determined by a prescribed maximum order $P$:
    $$
    N_t=\binom{D+P}{P}
    $$
2.  **Linear System ($A \cdot \mathbf{c} = \mathbf{Q}$):** Using $N_s > N_t$ sample points (e.g., Latin Hypercube Sampling), the coefficients are calculated using the **Least Squares Method**:

$$
\underbrace{
\begin{pmatrix}
\Psi_{\boldsymbol{\alpha}_0}(\boldsymbol{\xi}_1) & \cdots & \Psi_{\boldsymbol{\alpha}_{N_t-1}}(\boldsymbol{\xi}_1) \\
\vdots & \ddots & \vdots \\
\Psi_{\boldsymbol{\alpha}_0}(\boldsymbol{\xi}_{N_s}) & \cdots & \Psi_{\boldsymbol{\alpha}_{N_t-1}}(\boldsymbol{\xi}_{N_s})
\end{pmatrix}}_{A}
\begin{pmatrix}
c_0 \\
\vdots \\
c_{N_t-1}
\end{pmatrix}
=
\begin{pmatrix}
Q(\boldsymbol{\xi}_1) \\
\vdots \\
Q(\boldsymbol{\xi}_{N_s})
\end{pmatrix}
$$

---

## 3. Analytical Moment and Sensitivity Calculation

The orthogonality property allows for the direct and exact calculation of statistical properties from the coefficients $c_k$.

### A. Mean Value $\mathbb{E}[Q(\boldsymbol{\xi})]$

Since the zero-order basis function is $\Psi_{\boldsymbol{\alpha_0}}(\boldsymbol{\xi}) = 1$, and all higher-order terms have a zero expected value, the mean is simply the first coefficient:
$$
\mathbb{E}[Q(\boldsymbol{\xi})] = c_0
$$

### B. Total Variance $\mathrm{Var}[Q(\boldsymbol{\xi})]$

The variance is the sum of the squared coefficients multiplied by the squared norms of their corresponding basis functions (excluding the constant term $k=0$):
$$
\mathrm{Var}[Q(\boldsymbol{\xi})] = \sum_{k=1}^{N_t-1} c_k^2 \Vert \Psi_{\boldsymbol{\alpha_k}} \Vert^2
$$

### C. Sobol Indices ($S_I$)

The **partial variance** $V_I$ associated with any subset of input features $I \subseteq \{1, 2, \dots, D\}$ is calculated by summing the contributions of all basis functions $\Psi_{\boldsymbol{\alpha_k}}(\boldsymbol{\xi})$ whose multi-index $\boldsymbol{\alpha_k}$ is non-zero **only** for the dimensions in set $I$.

The **Sobol Index** $S_I$ is the ratio of the partial variance $V_I$ to the total variance $V$:

$$
S_I = \frac{V_I}{\mathrm{Var}[Q(\boldsymbol{\xi})]} = \frac{1}{\mathrm{Var}[Q(\boldsymbol{\xi})]} \sum_{\boldsymbol{\alpha_k} \in \mathcal{A}_I} c_k^2 \Vert \Psi_{\boldsymbol{\alpha_k}} \Vert^2
$$

where $\mathcal{A}_I = \{ \boldsymbol{\alpha_k} : \alpha_{i}^k \geq 1 \text{ if } i \in I \text{ and } \alpha_{j}^k = 0 \text{ if } j \notin I \}$.

**Example: Second-Order Joint Sobol Index $S_{1,2}$**

For $I=\{1, 2\}$, the partial variance $V_{1,2}$ sums the coefficients where $\boldsymbol{\alpha_k}=(\alpha_1^k, \alpha_2^k, 0, \dots, 0)$, with $\alpha_1^k \geq 1$ and $\alpha_2^k \geq 1$:
$$
S_{1,2} = \frac{1}{\mathrm{Var}[Q(\boldsymbol{\xi})]} \sum_{\substack{\boldsymbol{\alpha_k} \\ \alpha_1^k \geq 1, \alpha_2^k \geq 1 \\ \alpha_i^k = 0, i>2}} c_k^2 \Vert \Psi_{\boldsymbol{\alpha_k}} \Vert^2
$$
