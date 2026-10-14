# Symbolic regression configuration

Symbolic Lab reads its research settings directly from `results/SR_train.py`, without executing that script. The same configuration is used by the interface, worker and exported experiment records.

## Configuration presets

| Setting | Paper settings | Quick run |
|---|---|---|
| Expression template | `f(Po)*g(Ex)*h(Sk)*ksi(Sx)*eta(kavg)` | Same |
| Unary operators | `tanh`, `exp`, `square`, `cos` | Same |
| Binary operators | `+`, `-`, `*`, `/` | Same |
| Nested-operator limits | Original table below | Same |
| Maximum weighted complexity (`maxsize`) | 60 | 60 |
| Constant complexity (`complexity_of_constants`) | 3 | 3 |
| Variable / operator complexity | 1 / 1 | 1 / 1 |
| Iterations | 1500 | 25 |
| Populations | 90 | 4 |
| Population size | 80 | 80 |
| Cycles per iteration | 1200 | 100 |
| Elementwise loss | `L1DistLoss()` | Same |
| Precision | 64 | 64 |
| PySR model selection | `best` | Same |
| PySR execution | `multithreading` | `serial` |
| Search time limit | None | 60 seconds |

The paper preset reproduces the original script's explicit search settings. The quick preset changes the search budget and execution mode for interactive use. A fixed seed is provided by the interface; PySR serial mode is deterministic, while multithreading follows the paper's execution mode.

## Expression constraints

The **Expression constraints** tab contains the operator lists, complexity settings and editable nesting table. Rows identify an outer operator and columns identify an operator nested within it.

| Outer / inner | exp | tanh | square | cos |
|---|---:|---:|---:|---:|
| exp | 0 | 0 | 1 | 0 |
| tanh | 0 | 0 | 0 | 0 |
| cos | 0 | 0 | 1 | 0 |
| square | 1 | 0 | 0 | 0 |

A value of **0** forbids the inner operator beneath the outer operator. **1** permits it once along a branch. In a custom configuration, **−1** removes that particular limit. Only selected operators are passed to the search backend.

For example, the source configuration permits `exp(square(x))`, `cos(square(x))` and `square(exp(x))`. It excludes `exp(exp(x))`, `tanh(tanh(x))` and `square(square(x))`. These checks include operators separated by binary expressions, not just adjacent function calls.

Maximum complexity uses the specified weights: each constant contributes 3 by default, and each variable or operator contributes 1. Changing the constant weight changes which expressions satisfy `maxsize`.

## Search backends

**PySR** applies the configured template, nesting table and complexity weights through `PySRRegressor`. Its population count, cycles and execution mode are available in **Search budget**. A time limit of zero disables the limit, matching the original script.

**Python GP** applies the same selected nesting limits and constant weight in its own candidate checks. It uses a single population and protected numerical operators. Its fitted scale is included in the expression complexity; the fixed outer template structure is not charged as a searched subexpression. PySR-specific population counts, cycles and execution mode are disabled for this backend.

Both engines train on the 96 non-DNSLC cases. The eight DNSLC cases remain held out. Run records include the effective settings, equations and predictions; the process log also prints the PySR options passed to the library.

## Inspect and export

- **Prediction** shows training and held-out predictions.
- **Search trajectory** shows the live GP loss history or PySR complexity–loss front.
- **Equations & export** includes the final equation table, effective configuration and downloadable JSON/CSV files.
- **Live process log** shows initialization and search output.
- **View verified PySR demo** opens the included short-budget experiment with its saved settings and log.
