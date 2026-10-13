"""Explicit feature mappings and numerically continuous published model."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Po", "Ex", "Sk", "kavg", "Sx"]
LABELS = ["Porosity · Po", "Effective slope · Ex", "Skewness · Sk", "Mean height · kavg", "Streamwise slope · Sx"]
BOUNDS = [[.240, .929], [.158, 1.177], [-1.445, 2.519], [.007, .477], [.028, .750]]
PAPER_DISTRIBUTIONS = ["U", "N", "N", "U", "N"]
PAPER_PARAMETERS = [[.240, .929], [.517, .249], [.367, .841], [.007, .477], [.159, .107]]


def load_data():
    df = pd.read_csv(ROOT / "results/csvFiles/rough_surface_statistics.csv")
    rename = {"pro_ch": "Po", "ESx_ch": "Ex", "sk_ch": "Sk", "kbar_ch": "kavg"}
    df = df.rename(columns=rename)
    for col in FEATURES + ["ks", "krms_ch"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["target"] = df["ks"] / df["krms_ch"]
    df["split"] = np.where(df.label == "DNSLC", "Held-out · DNSLC", "Training")
    return df


def factors(x):
    """Same expression as run_analysis.py; continuous limits at Sk=0/kavg=0."""
    x = np.asarray(x, dtype=float)
    po, ex, sk, ka, sx = x.T
    ratio = np.divide(sk, np.tanh(sk), out=np.ones_like(sk), where=np.abs(sk) > 1e-10)
    inv = np.divide(-.024, ka**2, out=np.full_like(ka, -np.inf), where=ka != 0)
    return np.column_stack([
        (np.cos(po) - po*np.exp(po))**2 + 8.331,
        np.tanh((ex-.084)/.294),
        np.tanh(sk) + ratio,
        2*ka - .692 - np.tanh(inv)*np.exp(ka),
        np.exp(2*sx),
    ])


def drag_model(x):
    return np.prod(factors(x), axis=1)


def ishigami(x):
    x = np.asarray(x, dtype=float)
    return np.sin(x[:, 0]) + 7*np.sin(x[:, 1])**2 + .1*x[:, 2]**4*np.sin(x[:, 0])


def metrics(y, pred):
    y, pred = np.asarray(y).ravel(), np.asarray(pred).ravel()
    ss = np.sum((y-y.mean())**2)
    rel = np.abs(pred-y) / np.maximum(np.abs(y), 1e-12)
    return {"MAE": float(np.mean(np.abs(pred-y))), "MARE (%)": float(100*rel.mean()),
            "Max ARE (%)": float(100*rel.max()), "R²": float(1-np.sum((pred-y)**2)/ss) if ss else float("nan")}


def published_predictions():
    df = load_data()
    df["prediction"] = drag_model(df[FEATURES].to_numpy())
    df["ARE (%)"] = 100*np.abs(df.prediction-df.target)/df.target
    return df
