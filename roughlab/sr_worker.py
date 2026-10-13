"""Isolated worker; a Julia runtime crash cannot bring down Streamlit."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roughlab.models import FEATURES, load_data, metrics


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_pysr(config, job):
    import numpy as np
    from pysr import PySRRegressor, TemplateExpressionSpec
    df = load_data()
    train = (df.label != "DNSLC").to_numpy()
    x, y = df[FEATURES].to_numpy(), df.target.to_numpy()
    # SR_train.py uses Po, Ex, Sk, Sx, kavg; reorder explicitly here.
    original_order = [0, 1, 2, 4, 3]
    x = x[:, original_order]
    names = [FEATURES[i] for i in original_order]
    kwargs = {}
    if config["template"]:
        kwargs["expression_spec"] = TemplateExpressionSpec(
            expressions=["f", "g", "h", "ksi", "eta"], variable_names=names,
            combine="f(Po)*g(Ex)*h(Sk)*ksi(Sx)*eta(kavg)")
    original_constraints = {
        "exp": {"exp":0,"tanh":0,"square":1,"cos":0},
        "tanh": {"tanh":0,"exp":0,"square":0,"cos":0},
        "cos": {"tanh":0,"exp":0,"square":1,"cos":0},
        "square": {"tanh":0,"exp":1,"square":0,"cos":0},
    }
    constraints = {a:{b:n for b,n in original_constraints[a].items() if b in config["unary"]} for a in config["unary"]}
    model = PySRRegressor(
        niterations=config["iterations"], populations=config["populations"],
        population_size=config["population"], maxsize=config["maxsize"],
        ncycles_per_iteration=config["cycles"], binary_operators=config["binary"],
        unary_operators=config["unary"], nested_constraints=constraints,
        elementwise_loss="L1DistLoss()", precision=64, model_selection="best",
        random_state=config["seed"], deterministic=True, parallelism="serial",
        timeout_in_seconds=config["timeout"], output_directory=str(job / "pysr"),
        progress=False, verbosity=1, complexity_of_constants=3, **kwargs)
    start = time.perf_counter()
    atomic_json(job/"progress.json", {"stage":"search", "engine":"PySR", "iterations":config["iterations"]})
    print(f"Starting PySR search: {config['iterations']} iterations, {config['populations']} populations, seed {config['seed']}",flush=True)
    model.fit(x[train], y[train], variable_names=names)
    pred = model.predict(x)
    frontier = model.equations_[["complexity", "loss", "equation"]].to_dict(orient="records")
    return {"engine": "PySR · live search", "equation": str(model.get_best()["equation"]),
            "frontier": frontier, "history": [], "predictions": pred.tolist(),
            "train_metrics": metrics(y[train],pred[train]), "test_metrics": metrics(y[~train],pred[~train]),
            "seconds": time.perf_counter()-start, "config": config}


def main(job):
    job = Path(job)
    config = json.loads((job/"config.json").read_text(encoding="utf-8"))
    atomic_json(job/"progress.json", {"stage":"initializing", "engine":config["engine"]})
    try:
        if config["engine"] == "PySR":
            result = run_pysr(config, job)
        else:
            from roughlab.symbolic import run_gp
            result = run_gp(config, lambda p:atomic_json(job/"progress.json",p))
        atomic_json(job/"result.json", result)
        atomic_json(job/"progress.json", {"stage":"complete"})
        print("Search complete. Held-out DNSLC samples were excluded from training.", flush=True)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        atomic_json(job/"error.json", {"error":error})
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1])
