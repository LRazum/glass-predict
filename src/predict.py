"""
predict.py — load saved models and predict properties (with uncertainty) for a
new glass composition. This is the function the future app will call.

Usage (CLI):
    python predict.py --comp "P=50,Na=35,Al=15"
    python predict.py --comp "P=40,Fe=20,Na=35,K=5" --target Tg
    python predict.py --csv new_compositions.csv         # one row per glass

Usage (import):
    from predict import predict_all, predict_one
    predict_all({"P": 50, "Na": 35, "Al": 15})

Compositions are given in mol% (any positive units work; they are renormalised).
For DC conductivity (log-modelled) the prediction is reported in S/cm with an
asymmetric 95% interval, since the model is linear in log10 space.
"""
from __future__ import annotations
import os
import sys
import glob
import argparse
import joblib
import numpy as np
import pandas as pd

import glass_pipeline_v2 as core

MODELS_DIR = os.environ.get("MODELS_DIR", "models")


def _load_models(target=None):
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "model_*.joblib")))
    if not paths:
        sys.exit(f"No models found in '{MODELS_DIR}/'. Run train.py first.")
    out = {}
    for p in paths:
        payload = joblib.load(p)
        out[payload["target"]] = payload
    if target is not None:
        if target not in out:
            sys.exit(f"No model for '{target}'. Available: {list(out)}")
        return {target: out[target]}
    return out


def _featurize(comp_rows: list[dict]) -> pd.DataFrame:
    """Turn raw composition dicts into the full physics-feature frame, matching
    exactly what was used at training time."""
    df = pd.DataFrame([{c: float(r.get(c, 0.0)) for c in core.COMPOSITION_COLS}
                       for r in comp_rows])
    return core.add_physics_features(df)


def predict_all(comp: dict, models=None) -> dict:
    """Predict every available property for one composition dict."""
    models = models or _load_models()
    feat = _featurize([comp])
    out = {}
    for target, payload in models.items():
        X = feat[payload["feature_cols"]]
        mean, std = payload["model"].predict(X, return_std=True)
        m, s = float(mean[0]), float(std[0])
        if payload["log_target"]:                      # back-transform from log10
            out[target] = {
                "value": 10 ** m, "unit": payload["unit"],
                "ci95": [10 ** (m - 1.96 * s), 10 ** (m + 1.96 * s)],
                "log10_value": m, "log10_std": s,
                "model": payload["model_name"],
            }
        else:
            out[target] = {
                "value": m, "unit": payload["unit"],
                "ci95": [m - 1.96 * s, m + 1.96 * s], "std": s,
                "model": payload["model_name"],
            }
    return out


def predict_one(comp: dict, target: str) -> dict:
    return predict_all(comp, models=_load_models(target))[target]


def _print(comp, preds):
    print("\ncomposition (mol%):", ", ".join(f"{k}={v}" for k, v in comp.items() if v))
    print("-" * 64)
    for t, r in preds.items():
        lo, hi = r["ci95"]
        if "log10_value" in r:
            print(f"{t:<22} {r['value']:.3e} {r['unit']:<7} "
                  f"[95%: {lo:.2e} – {hi:.2e}]  ({r['model']})")
        else:
            print(f"{t:<22} {r['value']:>8.2f} {r['unit']:<7} "
                  f"[95%: {lo:.2f} – {hi:.2f}]  ({r['model']})")


def _parse_comp(s: str) -> dict:
    d = {}
    for part in s.split(","):
        k, v = part.split("=")
        d[k.strip()] = float(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", help='e.g. "P=50,Na=35,Al=15"')
    ap.add_argument("--csv", help="CSV with one composition per row")
    ap.add_argument("--target", help="predict only this property")
    args = ap.parse_args()
    models = _load_models(args.target)

    if args.csv:
        df = pd.read_csv(args.csv)
        rows = df.to_dict("records")
        for row in rows:
            comp = {c: row.get(c, 0.0) for c in core.COMPOSITION_COLS}
            _print(comp, predict_all(comp, models))
    else:
        comp = _parse_comp(args.comp) if args.comp else {"P": 50, "Na": 35, "Al": 15}
        if not args.comp:
            print("[no --comp given; using demo composition P50-Na35-Al15]")
        _print(comp, predict_all(comp, models))


if __name__ == "__main__":
    main()
