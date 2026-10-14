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
from roughlab.sr_config import PAPER_TEMPLATE, normalize_config, pysr_options


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            # Windows may briefly hold a reader's handle without delete sharing.
            if attempt==9: raise
            time.sleep(.02*(attempt+1))


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
        kwargs["expression_spec"] = TemplateExpressionSpec(**PAPER_TEMPLATE)
    options=pysr_options(config,job/"pysr")
    print("Effective PySR settings:\n"+json.dumps(options,indent=2),flush=True)
    model = PySRRegressor(**options, **kwargs)
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
    config = normalize_config(json.loads((job/"config.json").read_text(encoding="utf-8")))
    atomic_json(job/"config.json",config)
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
