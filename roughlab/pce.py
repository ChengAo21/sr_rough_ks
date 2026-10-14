"""Run the repository's PolyChaos class, with independent validation samples."""
import contextlib
import io
import sys
import time
import math
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .models import ROOT, FEATURES, drag_model, ishigami, factors

sys.path.insert(0, str(ROOT / "code_files"))
from pce_core import PolyChaos


def fit_pce(case, order, samples, seed, distrib, parameters, callback=None):
    dim = len(distrib)
    nt = math.comb(dim + order, order)
    if samples < nt:
        raise ValueError(f"Need at least {nt} samples for {nt} basis terms.")
    for dist, p in zip(distrib, parameters):
        if dist not in ("U", "N") or not np.isfinite(p).all():
            raise ValueError("Use finite parameters and distribution U or N.")
        if (dist == "U" and p[1] <= p[0]) or (dist == "N" and p[1] <= 0):
            raise ValueError("Uniform: upper > lower. Normal: standard deviation > 0.")
    start = time.perf_counter()
    fun = drag_model if case == "Roughness drag" else ishigami
    log = io.StringIO()
    with contextlib.redirect_stdout(log), threadpool_limits(limits=2):
        model = PolyChaos(order=order, distrib=distrib, param=parameters)
        if callback: callback(.15, f"Created {model.nt} polynomial basis terms")
        train = model.create_latin_hypercube_samples(samples, rd_state=seed)
        y = fun(train)
        model.regression_fit(train, y)
        model.norm_fit()
        if callback: callback(.65, "Least-squares fit complete; validating on independent LHS points")
        valid = model.create_latin_hypercube_samples(1200, rd_state=seed+1009)
        truth, pred = fun(valid), model.evaluate(valid)
        variance = model.coeff[1:]**2 * np.array([1/model.norm_factor(a) for a in model.multi_index[1:]])
        total = variance.sum()
        first, totals = [], []
        for j in range(dim):
            active = model.multi_index[1:] != 0
            first.append(float(variance[active[:, j] & (active.sum(axis=1)==1)].sum()/total) if total else 0.)
            totals.append(float(variance[active[:, j]].sum()/total) if total else 0.)
        history = []
        for degree in range(order+1):
            partial = np.zeros(len(valid))
            keep = model.multi_index.sum(axis=1) <= degree
            # Evaluate an exact truncation of the fitted expansion, not a fabricated trajectory.
            old = model.coeff.copy()
            model.coeff[~keep] = 0
            partial = model.evaluate(valid)
            model.coeff = old
            history.append({"degree": degree, "relative_l2": float(np.linalg.norm(partial-truth)/np.linalg.norm(truth))})
    if callback: callback(1., "Independent validation and Sobol decomposition complete")
    return {"model": model, "case": case, "order": order, "samples": samples, "seed": seed,
            "distrib": distrib, "parameters": parameters, "train": train, "y": y,
            "valid": valid, "truth": truth, "prediction": pred, "first": first, "total": totals,
            "interaction": float(1-sum(first)), "relative_l2": history[-1]["relative_l2"],
            "history": history, "seconds": time.perf_counter()-start, "log": log.getvalue()}


def coefficient_table(result):
    m = result["model"]
    data = pd.DataFrame(m.multi_index, columns=FEATURES if m.dim == 5 else ["x1", "x2", "x3"])
    data["coefficient"] = m.coeff
    data["degree"] = m.multi_index.sum(axis=1)
    return data


def mean_factor(model,feature,quadrature_order=128):
    """Mean of one analytical subfunction under its independent input measure."""
    a,b=model.param[feature]
    if model.distrib[feature].upper()=="U":
        nodes,weights=np.polynomial.legendre.leggauss(quadrature_order)
        values=(b-a)*nodes/2+(a+b)/2
        weights=weights/2
    else:
        nodes,weights=np.polynomial.hermite_e.hermegauss(quadrature_order)
        values=a+b*nodes
        weights=weights/np.sqrt(2*np.pi)
    points=np.zeros((len(nodes),5),dtype=float)
    points[:,feature]=values
    return float(np.dot(weights,factors(points)[:,feature]))


def pairwise_projection_coefficients(model,axes=(1,2)):
    """Marginalize the full PCE and remove the mean contribution of other factors.

    Independence and orthogonality eliminate every basis term with a nonzero
    index in the marginalized dimensions. Dividing by their analytical mean
    factor product gives a surrogate for the two-factor product itself.
    """
    if model.dim!=5 or len(set(axes))!=2 or not set(axes).issubset(range(5)):
        raise ValueError("Pairwise projection requires two distinct roughness inputs.")
    other=[i for i in range(5) if i not in axes]
    scale=np.prod([mean_factor(model,i) for i in other])
    if not np.isfinite(scale) or abs(scale)<1e-12:
        raise ValueError("The other subfunctions have a zero or non-finite mean product; choose different input distributions.")
    keep=np.all(model.multi_index[:,other]==0,axis=1)
    return model.coeff*keep/scale
