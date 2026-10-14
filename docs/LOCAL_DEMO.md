# Local setup and demonstration guide

The application entry point is `streamlit_app.py`. The default local address is **http://127.0.0.1:8507**.

The web app is **[Roughness Lab on Streamlit](https://jfe-rough-sr.streamlit.app/)**. For Community Cloud deployment, use **Python 3.12** and follow the [deployment guide](STREAMLIT_DEPLOYMENT.md).

## Start the application

From the repository root, run:

```powershell
.\start_streamlit.ps1
```

The Windows launcher creates a virtual environment and installs the base dependencies on first use. Add `-SkipInstall` to reuse an installed environment, or `-Port 8508` if the default port is occupied. To choose a Python installation explicitly:

```powershell
.\start_streamlit.ps1 -PythonPath C:\path\to\python.exe
```

To include the original PySR backend:

```powershell
.\start_streamlit.ps1 -PySR
```

For a standard Python environment on Windows, macOS or Linux:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py --server.port 8507
```

Install the optional PySR dependencies with:

```bash
python -m pip install -r requirements-pysr.txt
```

The overview, analytical model explorer, Python GP engine and PCE calculations work without Julia. PySR also needs Julia and SymbolicRegression.jl; its first run can take longer while dependencies initialize and precompile. Progress and initialization messages appear in the process log.

## Suggested demonstration

1. **Overview:** rotate the subfunction product g(Ex) × f(Sk) and inspect the research workflow and model predictions across five data families. The remaining subfunctions contribute one.
2. **Symbolic Lab:** select **Quick run** for a short experiment, or **Paper settings** to restore the original PySR configuration. The **Expression constraints** tab exposes the unary/binary operators, maximum weighted complexity, constant-complexity weight and nested-operator limits. Both presets use maxsize 60 and constant complexity 3. Select **Run symbolic search**, then inspect **Search trajectory** or download the configuration, equations and predictions. See [SYMBOLIC_REGRESSION.md](SYMBOLIC_REGRESSION.md) for the complete settings.
3. **PCE Studio:** use the default degree-4 expansion and 800 LHS samples with the uniform preset. Select **Fit PCE & analyze**, then **Play**. Open **Sensitivity** to compare first-order and total-order Sobol indices.
4. Select **Paper · Table 5 mixed** for the degree-6, 2500-sample paper preset. Changing controls leaves the previous completed result visible and clearly identified until you select **Fit PCE** again.
5. Select **Ishigami benchmark**, using its default degree-6 expansion, to compare the calculated sensitivity indices with standard reference values.
6. **Model Explorer:** vary the five geometric statistics and inspect their univariate subfunctions and pairwise products. Only the selected two subfunctions enter the pairwise surface; the orange marker shows their selected input values.

In a fresh Symbolic Lab session, **View verified PySR demo** opens a previously completed experiment. It is labelled **SAVED VERIFIED RUN** to distinguish it from a new search.

## Interpreting the results

- Recomputing the public expression on the 104 supplied cases gives **9.5978895%** mean absolute relative error, **30.7923720%** maximum absolute relative error and **0.93515384** R².
- Python GP is a lightweight genetic-programming search using the selected operators, nesting limits and constant-complexity weight, with protected numerical operators. It is an educational backend distinct from PySR.
- All eight DNSLC cases are excluded from training. Input mappings and the target `ks/krms_ch` are explicit in `roughlab/models.py`.
- PCE calls the original `PolyChaos` class in `code_files/pce_core.py`. Inputs are assumed independent, and the mixed preset follows Table 5 of the paper.
- **Validation relative L₂** uses 1200 independent LHS points. Untruncated normal inputs can extend beyond the empirical feature range, so this diagnostic is separate from the paper's surrogate evaluation on its original training cases.
- Roughness response surfaces show products of two subfunctions, with all remaining subfunctions contributing one. The animation projects the full-model PCE onto g(Ex) × f(Sk), removes the other factors' mean contribution, and retains progressively higher degrees. Full-model statistics and sensitivity indices are calculated independently of this visualization.
- Settings, logs, coefficients, sensitivity indices and predictions are available for inspection or export.

## Verification and saved experiments

Browser verification covers both live search backends, PCE fitting, the animation's Play control, sensitivity plots and all five pages. Numerical verification checks data mappings, published-model metrics, continuous limits, exact polynomial moments under uniform and normal inputs, Sobol interactions, invalid settings and exclusion of test targets from training. A separate application check covers all pages and the Ishigami fit.

```bash
python -m unittest discover -s tests -v
python tools/build_assets.py
```

Saved evidence:

- `verification-results.json`: model metrics and the PCE experiment settings.
- `browser-verification.json`: browser verification of the main interface.
- `pysr-browser-verification.json`: browser verification of the PySR run.
- `runs/pysr_seed42.json` and `runs/pysr_process.txt`: a completed short-budget PySR search and its log.
- `runs/python_gp_seed42.json`: a completed Python GP search.
- `runs/tested-versions.json`: Python package versions used during verification.
- `../assets/screenshots/`: English-language screenshots of the application.

The supplied PySR demonstration uses the **Quick run** preset: 25 iterations, 4 populations, population size 80, maxsize 60, constant complexity 3, the original nesting table, 100 cycles per iteration and seed 42. Its metrics describe a short exploratory run. The complete effective configuration is included in the saved JSON and process log.

The numerical research implementation is preserved; source comments have been translated to English. The presentation adapter adds continuous limits at `Sk=0` and `kavg=0`. Figures and animations are computed from code and data rather than cropped from the paper.

## Julia initialization troubleshooting

PySR runs in an isolated worker process. If initialization fails, inspect the process log for Julia version, registry, network or package-installation errors. You can cancel the task and continue using Python GP, the model explorer and PCE.

On the development machine, Windows Julia's schannel downloader reported `SEC_E_NO_CREDENTIALS`. The optional helper `tools/julia_pkg_proxy.py` bridges official Julia package requests through Python's verified HTTPS client and binds only to `127.0.0.1`. Julia continues to verify package and artifact hashes; certificate verification remains enabled. This helper is only needed for that installation issue and can be stopped afterward. A normal installation does not require it.

The development installation completed successfully with PySR 2.7.0 and its required Julia dependencies. None of those machine-specific caches are included in the release archive.

## Publish the repository

Extract the release ZIP, then copy the contents of its `JFE_rough_codes` folder into your Git working directory. The archive contains the application, original numerical code and notebooks, input data, figures, screenshots, documentation, citation files, dependency specifications and license.

Local environments, Julia/Python caches, temporary jobs, Git metadata, previous release archives and the supplied reference PDFs are excluded. `.gitignore` prevents local runtime files from being committed. See `CITATION_VERIFICATION.md` for the verified bibliographic metadata and `STREAMLIT_DEPLOYMENT.md` for publishing the web app.

To rebuild the release archive:

```bash
python tools/build_release.py
```
