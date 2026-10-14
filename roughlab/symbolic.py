"""A small, real genetic-programming search for instant, Julia-free demos.

It is deliberately labelled separately from the paper's PySR backend. It searches
the same five inputs and operator family; it does not replay stored equations.
"""
import math
import time
import numpy as np
from .models import FEATURES, load_data, metrics
from .sr_config import normalize_config


def complexity(t,constant_complexity=3):
    if t[0]=="c": return constant_complexity
    if t[0]=="v": return 1
    return 1+sum(complexity(a,constant_complexity) for a in t[1:])


def respects_nesting(tree,constraints):
    """Match SymbolicRegression.jl's maximum nesting along any branch."""
    def count(node,operator):
        if node[0] in ("v","c"): return 0
        return int(node[0]==operator)+max(count(child,operator) for child in node[1:])
    if tree[0] in ("v","c"): return True
    for inner,limit in constraints.get(tree[0],{}).items():
        if max(count(child,inner) for child in tree[1:])>limit: return False
    return all(respects_nesting(child,constraints) for child in tree[1:])


def evaluate(t, x):
    op = t[0]
    if op == "v": return x[:, t[1]]
    if op == "c": return np.full(len(x), t[1])
    a = evaluate(t[1], x)
    with np.errstate(all="ignore"):
        if op == "exp": return np.exp(np.clip(a, -15, 15))
        if op == "tanh": return np.tanh(a)
        if op == "cos": return np.cos(a)
        if op == "square": return np.clip(a, -1e6, 1e6)**2
        b = evaluate(t[2], x)
        if op == "+": return a+b
        if op == "-": return a-b
        if op == "*": return np.clip(a*b, -1e12, 1e12)
        if op == "/": return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b)>1e-8)
    raise ValueError(op)


def expression(t):
    if t[0] == "v": return FEATURES[t[1]]
    if t[0] == "c": return f"{t[1]:.5g}"
    if len(t) == 2: return f"{t[0]}({expression(t[1])})"
    return f"({expression(t[1])} {t[0]} {expression(t[2])})"


def run_gp(config, progress=None):
    config=normalize_config(config)
    rng = np.random.default_rng(config["seed"])
    df = load_data()
    train = (df.label != "DNSLC").to_numpy()
    x = df[FEATURES].to_numpy()
    y = df.target.to_numpy()
    xt, yt = x[train], y[train]
    unary, binary = config["unary"], config["binary"]
    template = config["template"]
    popsize, generations = config["population"], config["iterations"]
    maxsize = config["maxsize"]
    constant_cost=config["constant_complexity"]
    constraints=config["nested_constraints"]
    frontier = {}
    history = []
    best = None
    start = time.perf_counter()

    def random_tree(depth, var=None):
        if depth <= 0 or rng.random() < .32:
            return ("v", int(rng.integers(5)) if var is None else var) if rng.random() < .78 else ("c", float(rng.uniform(-1.5, 2)))
        if unary and (not binary or rng.random() < .45):
            return (str(rng.choice(unary)), random_tree(depth-1, var))
        if binary:
            return (str(rng.choice(binary)), random_tree(depth-1, var), random_tree(depth-1, var))
        return random_tree(0, var)

    def make():
        return tuple(random_tree(2, j) for j in range(5)) if template else random_tree(3)

    def mutate(t, var=None):
        if rng.random() < .3 or t[0] in ("v", "c"): return random_tree(2, var)
        index = int(rng.integers(1, len(t)))
        out = list(t)
        out[index] = mutate(t[index], var)
        return tuple(out)

    def score(individual):
        nonlocal best
        trees=individual if template else (individual,)
        if not all(respects_nesting(t,constraints) for t in trees): return math.inf,None
        base_size=sum(complexity(t,constant_cost) for t in trees)
        if base_size>maxsize: return math.inf,None
        raw = np.prod(np.column_stack([evaluate(t, xt) for t in individual]), axis=1) if template else evaluate(individual, xt)
        if not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1e13: return math.inf, None
        # For a product template, a fitted scale can be absorbed in one subfunction.
        if template:
            mask = np.abs(raw) > 1e-10
            scale = float(np.median(yt[mask]/raw[mask])) if mask.any() else 0.
            offset = 0.
        else:
            a = np.column_stack([raw, np.ones(len(raw))])
            scale, offset = np.linalg.lstsq(a, yt, rcond=None)[0]
        # The product template has fixed outer structure. A fitted scale is
        # absorbed into one factor; an affine intercept exists only without it.
        size=base_size+constant_cost+1
        if not template: size+=constant_cost+1
        if size>maxsize: return math.inf,None
        pred = scale*raw+offset
        loss = float(np.mean(np.abs(pred-yt)))
        equation = " * ".join(expression(t) for t in individual) if template else expression(individual)
        rendered=f"{scale:.7g} * ({equation})"+(f" + {offset:.7g}" if not template else "")
        record = {"complexity": size, "loss": loss, "equation": rendered,
                  "tree": individual, "scale": float(scale), "offset": float(offset)}
        if size not in frontier or loss < frontier[size]["loss"]: frontier[size] = record
        if best is None or loss < best["loss"]: best = record
        return loss + config["parsimony"]*size, record

    population = [make() for _ in range(popsize)]
    # Plain variable-only starts, not the published model or pre-trained equations.
    population[0] = tuple(("v", j) for j in range(5)) if template else ("v", 1)
    for generation in range(generations):
        ranked = sorted([(score(ind)[0], i, ind) for i, ind in enumerate(population)], key=lambda a:a[0])
        elites = [entry[2] for entry in ranked[:max(3, popsize//8)]]
        history.append({"generation": generation+1, "loss": best["loss"], "complexity": best["complexity"]})
        if progress: progress({"stage": "search", "generation": generation+1, "total": generations, "loss": best["loss"]})
        new = elites.copy()
        while len(new) < popsize:
            if rng.random() < .18:
                child = make()
            else:
                parent = elites[int(rng.integers(len(elites)))]
                if template:
                    child = list(parent)
                    j = int(rng.integers(5))
                    child[j] = mutate(child[j], j)
                    child = tuple(child)
                else: child = mutate(parent)
            new.append(child)
        population = new
        if config["timeout"] and time.perf_counter()-start > config["timeout"]: break
    records, running = [], math.inf
    for size, record in sorted(frontier.items()):
        if record["loss"] < running:
            running = record["loss"]
            records.append({k:record[k] for k in ("complexity", "loss", "equation")})
    raw = np.prod(np.column_stack([evaluate(t, x) for t in best["tree"]]), axis=1) if template else evaluate(best["tree"], x)
    pred = best["scale"]*raw + best["offset"]
    return {"engine": "Python GP · live search", "equation": best["equation"], "frontier": records,
            "history": history, "predictions": pred.tolist(), "train_metrics": metrics(y[train], pred[train]),
            "test_metrics": metrics(y[~train], pred[~train]), "seconds": time.perf_counter()-start,
            "config": config, "completed_generations": len(history),
            "structure":{"factors":best["tree"]} if template else {"tree":best["tree"]},
            "scale":best["scale"],"offset":best["offset"]}
