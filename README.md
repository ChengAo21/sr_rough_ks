<div align="center">

![Roughness Lab — rough surfaces, readable physics](assets/banner.svg)

**Feature importance-informed symbolic drag modelling, from discovery to interpretation.**

[![Paper · JFE 2026](https://img.shields.io/badge/JFE_2026-10.1115%2F1.4070838-087f8c?style=flat-square)](https://doi.org/10.1115/1.4070838)
[![Streamlit app](https://img.shields.io/badge/Streamlit-Open_Roughness_Lab-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://jfe-rough-sr.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-536e70?style=flat-square)](#run-locally)
[![License: MIT](https://img.shields.io/badge/License-MIT-e58b53?style=flat-square)](LICENSE)

[Research](#research-overview) · [Model performance](#model-performance) · [Web app](https://jfe-rough-sr.streamlit.app/) · [Local setup](#run-locally) · [PCE & sensitivity](#uncertainty-and-sensitivity) · [Related publication](#related-publication) · [Cite](#cite-this-work)

</div>

## Research overview

Predicting drag across diverse rough surfaces calls for a model that is both accurate and readable. This work identifies influential geometric features, learns a structured analytical expression for equivalent sandgrain roughness, and uses polynomial chaos expansion (PCE) to interpret the model's sensitivity.

**Ao Cheng, Zhideng Zhou, Xiaolei Yang and Guowei He**  
*A Drag Model for Rough Surfaces Learned Using Feature Importance-Informed Symbolic Regression*  
**Journal of Fluids Engineering 148(4), 041302 (April 2026).** [Read the article ↗](https://doi.org/10.1115/1.4070838)

| Training surfaces | Unseen test surfaces | Overall mean relative error | Overall maximum relative error |
|:---:|:---:|:---:|:---:|
| **96** | **8** | **9.60%** | **30.79%** |

The paper reports these results across the 104 supplied cases. Evaluating the public expression on the repository data reproduces 9.5979% mean and 30.7924% maximum absolute relative error.

![Research workflow: data, feature importance, symbolic regression and sensitivity analysis](assets/workflow.svg)

Five geometric inputs form a product of univariate subfunctions:

$$
\frac{k_s}{k_{\mathrm{rms}}}=h(P_o)\,g(E_x)\,f(S_k)\,\psi(k_{\mathrm{avg}})\,\phi(S_x).
$$

The multiplicative template makes each feature's contribution available for inspection. Sections 4.4–4.5 and Appendix C of the article describe the sensitivity analysis and PCE formulation.

## Model performance

![Model predictions and errors by data family, recomputed from the public expression](assets/model_results.png)

Each point evaluates the public analytical expression. DNSLC diamonds mark the eight held-out cases; the second panel summarizes errors within each data family. The plots are newly drawn from the repository data.

The application also displays the two archived symbolic-search fronts in `results/csvFiles/`, separately from newly executed searches.

## Explore Roughness Lab

**[Open the Streamlit app ↗](https://jfe-rough-sr.streamlit.app/)**

Explore the model in your browser, run a symbolic search, and watch a polynomial surrogate take shape.

| Workspace | What you can explore |
|:---|:---|
| **Overview** | Rotate the 3D response surface and inspect predictions across five data families. |
| **Symbolic Lab** | Set operators, seed, search budget and complexity; inspect equations, process logs, training metrics and held-out predictions. |
| **PCE Studio** | Configure independent uniform/normal input distributions, polynomial degree and LHS size; fit `PolyChaos`, play the expansion animation and inspect Sobol indices. |
| **Model Explorer** | Adjust all five features and examine their individual factors, joint response and source data. |
| **Research & Cite** | Read the publications, copy BibTeX and review the experiment settings. |

![A symbolic-regression experiment in the Streamlit interface](assets/screenshots/03-symbolic-result.png)

A completed PySR experiment is included as a [saved run](docs/runs/pysr_seed42.json), [process log](docs/runs/pysr_process.txt) and [result screenshot](assets/screenshots/10-pysr-result.png). Select **View verified PySR demo** to inspect it without starting another search.

## Run locally

Python **3.12** is the recommended version and was used for local verification.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py --server.port 8507
```

Open [http://localhost:8507](http://localhost:8507). On Windows, the launcher handles environment creation and dependency installation:

```powershell
.\start_streamlit.ps1
```

See the [setup and demonstration guide](docs/LOCAL_DEMO.md) for a walkthrough, or the [Streamlit deployment guide](docs/STREAMLIT_DEPLOYMENT.md) to deploy your own instance.

### Symbolic regression engines

**Python GP** provides a lightweight genetic-programming search for immediate exploration without Julia. It uses the same inputs and operator family, an optional separated-variable product template and protected numerical operators.

**PySR** adapts the original `results/SR_train.py`, retaining the five-factor template, float64 precision and L1 objective. To enable live PySR searches locally:

```bash
python -m pip install -r requirements-pysr.txt
```

Then select **PySR** in the app. The first run initializes Julia and SymbolicRegression.jl. Searches run in separate processes, expose their logs and can be cancelled.

The paper configuration uses **1500 iterations, 90 populations, population size 80, 1200 cycles per iteration and maximum size 60**. The interface defaults to a smaller budget for exploration; results from those short searches are labelled separately from the published model.

## Uncertainty and sensitivity

![The fitted PCE expansion assembled by polynomial degree](assets/pce_expansion.gif)

The left panel shows a slice of the symbolic drag model. The right panel progressively retains higher-degree terms from one fitted PCE. Other inputs are fixed at Po=0.634, kavg=0.067 and Sx=0.159. The animation illustrates polynomial approximation rather than fluid evolution in time.

![PCE validation and Sobol sensitivity indices](assets/pce_results.png)

This experiment uses the paper's **Table 5 mixed distributions**, **degree 6**, **462 basis terms**, **2500 LHS fitting samples** and **seed 42**. Validation uses **1200 independent LHS points**, including distribution tails beyond the empirical feature bounds. Exact settings and results are available in [verification-results.json](docs/verification-results.json).

The app also offers a bounded uniform preset and an **Ishigami benchmark** for checking sensitivity-index convergence. Calculations call the repository's `PolyChaos` implementation; Sobol indices assume independent inputs.

## Learning resources

Helpful examples and documentation:

- **[PySR examples](https://pysr.ai/examples)** and **[API reference](https://ai.damtp.cam.ac.uk/pysr/api)**: symbolic regression, expression templates, operators and search settings.
- **[UQpy PCE gallery](https://uqpyproject.readthedocs.io/en/latest/auto_examples/surrogates/pce/)**: surrogate construction and sensitivity-analysis examples.
- **[Chaospy polynomial introduction](https://chaospy.readthedocs.io/en/master/user_guide/polynomial/introduction.html)**: orthogonal polynomial expansions and uncertainty quantification.

## Repository structure

```text
streamlit_app.py                 Streamlit application entry point
roughlab/                       Model adapters, visualizations and experiment workers
code_files/                     Original PCE research implementation
  pce_core.py                   PolyChaos: LHS, least squares, moments and Sobol indices
  pce_utils.py                  Polynomial and quadrature utilities
  run_analysis.py               Original model and analysis example
results/
  SR_train.py                   Original PySR training configuration
  csvFiles/                     104 roughness cases and archived search fronts
  subFunc_plot.ipynb             Factor and interaction notebook
  loss_plot.ipynb                Search-front notebook
assets/                         Figures, animation and application screenshots
docs/                           Setup, deployment, saved runs and citation metadata
tools/                          Figure generation, browser checks and release packaging
tests/                          Numerical and application verification
CITATION.bib / CITATION.cff      Publication metadata
requirements*.txt               Base dependencies and optional PySR backend
start_streamlit.ps1              Windows launcher
```

## Related publication

**[A physics-embedded symbolic wall model for large-eddy simulation of separated flows](https://doi.org/10.1016/j.jcp.2026.115311)**  
Ao Cheng, Xin-Lei Zhang, Zhideng Zhou, Xiaolei Yang and Guowei He.  
*Journal of Computational Physics* **567**, 115311 (2026).

A related publication on interpretable, physics-embedded symbolic wall modelling for separated-flow large-eddy simulation.

```bibtex
@article{cheng2026wall,
  title   = {A physics-embedded symbolic wall model for large-eddy simulation of separated flows},
  author  = {Cheng, Ao and Zhang, Xin-Lei and Zhou, Zhideng and Yang, Xiaolei and He, Guowei},
  journal = {Journal of Computational Physics},
  volume  = {567},
  pages   = {115311},
  year    = {2026},
  doi     = {10.1016/j.jcp.2026.115311},
  url     = {https://doi.org/10.1016/j.jcp.2026.115311}
}
```

## Cite this work

If you use this code or model, please cite the JFE article:

```bibtex
@article{cheng2026drag,
  title   = {A Drag Model for Rough Surfaces Learned Using Feature Importance-Informed Symbolic Regression},
  author  = {Cheng, Ao and Zhou, Zhideng and Yang, Xiaolei and He, Guowei},
  journal = {Journal of Fluids Engineering},
  volume  = {148},
  number  = {4},
  pages   = {041302},
  year    = {2026},
  month   = apr,
  doi     = {10.1115/1.4070838},
  url     = {https://doi.org/10.1115/1.4070838}
}
```

Both entries are available in [CITATION.bib](CITATION.bib). [CITATION.cff](CITATION.cff) supports GitHub's citation interface. Publication metadata is documented in [citation verification](docs/CITATION_VERIFICATION.md).

## Contributors

Research, original code and scientific interpretation: **Ao Cheng and coauthors**.

**OpenAI Codex** assisted with the Streamlit interface, code integration, figure generation, documentation and verification.

Released under the [MIT License](LICENSE).
