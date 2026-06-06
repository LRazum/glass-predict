"""
evaluate.py — honest evaluation report + figures for every target.

Writes:
    reports/metrics.json   machine-readable metrics for all candidates
    reports/metrics.csv     flat table
    reports/metrics.md      readable summary (grouped vs random R2, calibration, scramble)
    reports/figures/parity_<target>.png        OOF predicted vs actual (with 95% bars)
    reports/figures/calibration_<target>.png    reliability curve for the workhorse
    reports/figures/learning_curve_<target>.png (only if RUN_LC=1; slower)

All metrics are out-of-fold under leave-one-system-out GroupKFold. The 'random'
column is random K-fold on the same data — the gap to 'grouped' is the amount of
apparent skill that is really just within-series interpolation.

Run:
    python evaluate.py
    GLASS_DATA=data/raw/global_glass_dataset.csv GP_RESTARTS=8 python evaluate.py
    RUN_LC=1 python evaluate.py        # also draw learning curves
"""
from __future__ import annotations
import os
import csv
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

import glass_pipeline_v2 as core

REPORTS = os.environ.get("REPORTS_DIR", "reports")
FIGS = os.path.join(REPORTS, "figures")
RUN_LC = os.environ.get("RUN_LC", "0") == "1"


def _fmt(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def parity_plot(target, best, res, space, path):
    yt, yp = res["y_true"], res["oof_pred"]
    ys = res["oof_std"]
    plt.figure(figsize=(6, 6))
    if np.all(np.isfinite(ys)) and np.any(ys > 0):
        plt.errorbar(yt, yp, yerr=1.96 * ys, fmt="o", ms=4, alpha=0.7,
                     ecolor="lightgray", elinewidth=1, capsize=2, label="OOF ±95%")
    else:
        plt.scatter(yt, yp, s=18, alpha=0.7, label="OOF")
    lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="ideal")
    from sklearn.metrics import r2_score, mean_absolute_error
    plt.title(f"{target} — {best}\nGROUPED R²={r2_score(yt, yp):.3f}  "
              f"MAE={mean_absolute_error(yt, yp):.3f}")
    plt.xlabel(f"actual [{space}]"); plt.ylabel(f"predicted [{space}]")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()


def calibration_plot(target, best, res, path):
    yt, yp, ys = res["y_true"], res["oof_pred"], res["oof_std"]
    if not (np.all(np.isfinite(ys)) and np.any(ys > 0)):
        return False
    nominal = np.linspace(0.05, 0.99, 19)
    observed = []
    err = np.abs(yt - yp)
    for p in nominal:
        z = norm.ppf(0.5 + p / 2.0)        # two-sided coverage
        observed.append(np.mean(err <= z * ys))
    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="perfect")
    plt.plot(nominal, observed, "o-", ms=4, label=best)
    plt.title(f"Calibration — {target}\n(want points on the diagonal)")
    plt.xlabel("nominal coverage"); plt.ylabel("observed coverage")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()
    return True


def learning_curve(target, best, df, path):
    """OOF R² as more chemical SYSTEMS are made available for training."""
    from sklearn.metrics import r2_score
    X, y, cols, sub = core.prepare_xy(df, target)
    groups, _ = core.infer_groups(sub)
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    fracs = np.linspace(0.4, 1.0, 5)
    xs, ys = [], []
    yt_all = (np.log10(np.clip(y, 1e-30, None)) if core.TARGETS[target]["log"] else y)
    for fr in fracs:
        keep = set(uniq[: max(2, int(fr * len(uniq)))])
        mask = np.array([g in keep for g in groups])
        Xs, yss, gs = X[mask].reset_index(drop=True), y[mask], groups[mask]
        k = max(2, min(5, len(set(gs))))
        m, *_ = core.cv_oof(best, target, Xs, yss, gs, GroupKFold(n_splits=k))
        xs.append(int(mask.sum())); ys.append(m["R2"])
    plt.figure(figsize=(6, 4.5))
    plt.plot(xs, ys, "o-", ms=5)
    plt.title(f"Learning curve — {target} ({best})")
    plt.xlabel("# training samples"); plt.ylabel("grouped OOF R²")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()


def main():
    os.makedirs(FIGS, exist_ok=True)
    real = os.path.exists(core.DATA_PATH)
    df = core.add_physics_features(core.load_data())

    all_metrics, csv_rows = {}, []
    md = ["# Phosphate-glass model evaluation",
          "",
          f"- data source: **{'real: ' + core.DATA_PATH if real else 'SYNTHETIC (demo only)'}**",
          f"- features: {len(core.feature_columns('Tg'))} "
          f"({len(core.COMPOSITION_COLS)} composition + {len(core.PHYS_FEATURES)} physics)",
          "- all metrics are out-of-fold under leave-one-system-out GroupKFold",
          "- `PICP95` = coverage of the 95% interval (want ≈0.95); "
          "`MPIW` = mean interval width (sharpness, lower is better)",
          ""]

    for target in core.TARGETS:
        if target not in df.columns:
            continue
        r = core.compare_models(df, target)
        best = r["best"]
        all_metrics[target] = {
            "best": best, "n": r["n"], "n_systems": r["n_groups"], "space": r["space"],
            "scramble_R2": r["scramble"],
            "candidates": {nm: {"grouped": r["results"][nm]["grouped"],
                                "random": r["results"][nm]["random"]}
                           for nm in r["candidates"]},
        }

        md.append(f"## {target}  (n={r['n']}, systems={r['n_groups']}, units={r['space']})")
        md.append("")
        md.append("| model | grouped R² | random R² | MAE | RMSE | PICP95 | MPIW |")
        md.append("|---|---|---|---|---|---|---|")
        for nm in r["candidates"]:
            g, rd = r["results"][nm]["grouped"], r["results"][nm]["random"]
            star = " **★**" if nm == best else ""
            md.append(f"| {nm}{star} | {_fmt(g['R2'])} | {_fmt(rd['R2'])} | "
                      f"{_fmt(g['MAE'])} | {_fmt(g['RMSE'])} | "
                      f"{_fmt(g.get('PICP95'))} | {_fmt(g.get('MPIW'))} |")
            csv_rows.append({"target": target, "model": nm, "is_best": nm == best,
                             "grouped_R2": _fmt(g["R2"]), "random_R2": _fmt(rd["R2"]),
                             "MAE": _fmt(g["MAE"]), "RMSE": _fmt(g["RMSE"]),
                             "PICP95": _fmt(g.get("PICP95")), "MPIW": _fmt(g.get("MPIW"))})
        md.append("")
        md.append(f"- **workhorse:** `{best}` (chosen by grouped R²)")
        md.append(f"- **y-scramble grouped R²:** {r['scramble']['mean']:+.3f} "
                  f"± {r['scramble']['std']:.3f}  *(must be ≈0 / negative — proves "
                  f"it isn't fitting noise)*")
        md.append("")

        # figures (use the workhorse OOF)
        res = r["results"][best]
        parity_plot(target, best, res, r["space"],
                    os.path.join(FIGS, f"parity_{target_safe(target)}.png"))
        ok = calibration_plot(target, best, res,
                              os.path.join(FIGS, f"calibration_{target_safe(target)}.png"))
        md.append(f"![parity](figures/parity_{target_safe(target)}.png)")
        if ok:
            md.append(f"![calibration](figures/calibration_{target_safe(target)}.png)")
        if RUN_LC:
            learning_curve(target, best, df,
                           os.path.join(FIGS, f"learning_curve_{target_safe(target)}.png"))
            md.append(f"![learning curve](figures/learning_curve_{target_safe(target)}.png)")
        md.append("")

    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump(_to_json(all_metrics), f, indent=2)
    with open(os.path.join(REPORTS, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    with open(os.path.join(REPORTS, "metrics.md"), "w") as f:
        f.write("\n".join(md))
    print(f"Wrote metrics.json / metrics.csv / metrics.md and figures to '{REPORTS}/'")
    if not real:
        print("NOTE: evaluated on SYNTHETIC data — numbers are illustrative only.")


def target_safe(t):
    return t.replace(" ", "_").replace("/", "_").replace("°", "").replace("-", "")


def _to_json(obj):
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
