"""
train.py — fit the final model per target on ALL available data and persist it.

For each target it:
  1. asks glass_pipeline_v2.compare_models() which model wins under honest
     grouped CV (the workhorse),
  2. refits that model on the full (non-missing) dataset,
  3. saves a self-describing joblib payload to MODELS_DIR,
  4. writes models/manifest.json summarising every target.

The saved payload carries the feature list, the log-transform flag and the CV
metrics, so predict.py / the future app can load a model and use it without
re-deriving anything. Loading requires glass_pipeline_v2.py to be importable
(the custom estimator classes live there).

Run:
    python train.py
    GLASS_DATA=data/raw/global_glass_dataset.csv GP_RESTARTS=8 python train.py
"""
from __future__ import annotations
import os
import json
import datetime
import joblib
import numpy as np

import glass_pipeline_v2 as core

MODELS_DIR = os.environ.get("MODELS_DIR", "models")


def safe_name(target: str) -> str:
    return target.replace(" ", "_").replace("/", "_").replace("°", "").replace("-", "")


def jsonable(obj):
    """Recursively coerce numpy types so json.dump won't choke."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    real = os.path.exists(core.DATA_PATH)
    df = core.add_physics_features(core.load_data())

    manifest = {
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_source": core.DATA_PATH if real else "SYNTHETIC (fallback)",
        "data_is_real": bool(real),
        "data_rows": int(len(df)),
        "feature_count": len(core.feature_columns("Tg")),
        "targets": {},
    }

    print("\n" + "#" * 78)
    print("TRAINING FINAL MODELS" + ("" if real else "  [SYNTHETIC DATA — demo only]"))
    print("#" * 78)

    for target in core.TARGETS:
        if target not in df.columns:
            print(f"  [skip] {target}: column not present")
            continue

        cmp = core.compare_models(df, target)         # selects workhorse honestly
        best = cmp["best"]
        X, y, cols, sub = core.prepare_xy(df, target)
        model = core.make_model(best, target).fit(X, y)   # refit on ALL data

        path = os.path.join(MODELS_DIR, f"model_{safe_name(target)}.joblib")
        payload = {
            "model": model,
            "model_name": best,
            "target": target,
            "feature_cols": cols,
            "log_target": core.TARGETS[target]["log"],
            "unit": core.TARGETS[target]["unit"],
            "n_train": int(cmp["n"]),
            "n_systems": int(cmp["n_groups"]),
            "cv": jsonable({
                "grouped": cmp["results"][best]["grouped"],
                "random": cmp["results"][best]["random"],
                "scramble_R2_mean": cmp["scramble"]["mean"],
                "scramble_R2_std": cmp["scramble"]["std"],
            }),
            "config": {
                "composition_cols": core.COMPOSITION_COLS,
                "phys_features": core.PHYS_FEATURES,
                "formers": sorted(core.FORMERS),
            },
            "trained_at": manifest["trained_at"],
            "data_is_real": bool(real),
            "core_module": "glass_pipeline_v2",
        }
        joblib.dump(payload, path)

        g = cmp["results"][best]["grouped"]
        manifest["targets"][target] = {
            "model": best, "path": path, "n_train": int(cmp["n"]),
            "n_systems": int(cmp["n_groups"]),
            "grouped_R2": jsonable(g["R2"]), "grouped_MAE": jsonable(g["MAE"]),
            "PICP95": jsonable(g.get("PICP95", float("nan"))),
            "scramble_R2": jsonable(cmp["scramble"]["mean"]),
        }
        print(f"  [ok]  {target:<22} model={best:<11} "
              f"grouped R2={g['R2']:.3f}  MAE={g['MAE']:.3f}  -> {path}")

    with open(os.path.join(MODELS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved {len(manifest['targets'])} models + manifest.json to '{MODELS_DIR}/'")
    if not real:
        print("NOTE: trained on SYNTHETIC data. Point GLASS_DATA at your real CSV and "
              "re-run to get real models.")
    return manifest


if __name__ == "__main__":
    train_all()
