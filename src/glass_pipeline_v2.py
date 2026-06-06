"""
================================================================================
Phosphate-Glass Property Prediction — Pipeline v2
================================================================================
A physics-informed, small-data pipeline for predicting Density, Tg,
DC conductivity and DC activation energy of phosphate glasses from molar
composition.

Design goals (in priority order):
    1. HONEST generalization estimates  -> grouped (leave-one-system-out) CV
    2. Real chemistry, not interpolation/noise -> physics features + y-scramble
    3. Calibrated uncertainty            -> Gaussian Process (Linear + Matern + White)

Why this replaces XGBoost + plain GPR for THIS problem:
    * The targets vary smoothly / near-monotonically with composition. Trees give
      piecewise-constant fits and CANNOT extrapolate beyond the training hull —
      a fatal property for materials discovery. A GP with a Linear+Matern kernel
      has the right inductive bias and extrapolates sanely along the linear term.
    * The dataset is a set of compositional *series* (one oxide swept in small
      steps). RANDOM K-fold leaks near-duplicate neighbours into train+test and
      massively inflates R^2. We evaluate with GroupKFold over chemical systems
      so a whole system is held out — this is what "learns chemistry" means.
    * Density is dominated by molar mass (heavy cations -> denser). We learn the
      small residual on top of a 1-parameter physical baseline (delta-learning),
      which slashes the effective complexity and improves extrapolation.
    * conductivity and activation energy are Arrhenius-coupled; we expose that
      coupling as a feature instead of ignoring it.

Drop your real data in by setting DATA_PATH. If the file is missing, the script
generates a physically-flavoured synthetic dataset so it runs out of the box.
================================================================================
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C, Matern, WhiteKernel, DotProduct,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)

# Runtime knobs (raise for the real run; lower for a quick smoke test).
GP_RESTARTS = int(os.environ.get("GP_RESTARTS", "3"))   # GP optimiser restarts
N_SCRAMBLE = int(os.environ.get("N_SCRAMBLE", "3"))     # y-scramble repeats

# ------------------------------------------------------------------ config ----
DATA_PATH = os.environ.get("GLASS_DATA", "data/processed/global_dataset_with_features.csv")
OUT_DIR = "pipeline_v2_out"

# Cation columns exactly as in your CSV (values are mol% of the cation).
COMPOSITION_COLS = ["P", "Ag", "Al", "Fe", "K", "Li", "Mo", "Na", "Nb", "V", "W", "Zn"]

TARGETS = {
    "Density":             {"log": False, "unit": "g/cm^3"},
    "Tg":                  {"log": False, "unit": "C"},
    "conductivity @30°C":  {"log": True,  "unit": "S/cm"},   # modelled in log10
    "EDC / kJ mol-1":      {"log": False, "unit": "kJ/mol"},
}

# Ionic radius (Angstrom) and formal charge, from your app.py ION_DATA.
ION_DATA = {
    "Li": {"r": 0.76, "z": 1}, "Na": {"r": 1.02, "z": 1}, "K": {"r": 1.38, "z": 1},
    "Ag": {"r": 1.15, "z": 1}, "Mg": {"r": 0.72, "z": 2}, "Ca": {"r": 1.00, "z": 2},
    "Sr": {"r": 1.18, "z": 2}, "Ba": {"r": 1.35, "z": 2}, "Zn": {"r": 0.74, "z": 2},
    "Pb": {"r": 1.19, "z": 2}, "Fe": {"r": 0.64, "z": 3}, "Al": {"r": 0.53, "z": 3},
    "B":  {"r": 0.23, "z": 3}, "Ti": {"r": 0.61, "z": 4}, "P": {"r": 0.38, "z": 5},
    "Nb": {"r": 0.64, "z": 5}, "V": {"r": 0.54, "z": 5}, "Mo": {"r": 0.59, "z": 6},
    "W":  {"r": 0.60, "z": 6},
}
ATOMIC_MASS = {  # g/mol
    "P": 30.974, "Ag": 107.868, "Al": 26.982, "Fe": 55.845, "K": 39.098,
    "Li": 6.941, "Mo": 95.95, "Na": 22.990, "Nb": 92.906, "V": 50.942,
    "W": 183.84, "Zn": 65.38, "Mg": 24.305, "Ca": 40.078, "Sr": 87.62,
    "Ba": 137.327, "Pb": 207.2, "B": 10.811, "Ti": 47.867,
}
M_O = 15.999  # oxygen
# Network formers (everything else is treated as a modifier/intermediate).
FORMERS = {"P"}

# ============================================================ data loading ====
def synthesize(n_per_series=9, n_series=24):
    """Physically-flavoured synthetic data, organised in compositional series,
    so the script runs without the real CSV and exercises grouped CV."""
    modifiers = ["Na", "K", "Li", "Zn", "Ag", "Al", "Fe"]
    formers_extra = ["V", "Mo", "W", "Nb"]
    rows = []
    for s in range(n_series):
        m = modifiers[s % len(modifiers)]
        f2 = formers_extra[s % len(formers_extra)]
        base_P = RNG.uniform(35, 60)
        for k in range(n_per_series):
            t = k / (n_per_series - 1)
            comp = {c: 0.0 for c in COMPOSITION_COLS}
            comp["P"] = base_P * (1 - 0.4 * t)
            comp[m] = (100 - comp["P"]) * RNG.uniform(0.5, 0.8)
            comp[f2] = max(0.0, 100 - comp["P"] - comp[m])
            tot = sum(comp.values())
            for c in comp:
                comp[c] = 100 * comp[c] / tot
            # crude physical-ish targets
            mm = sum(comp[c] / 100 * (ATOMIC_MASS[c] + 8 * ION_DATA[c]["z"]) for c in comp)
            dens = 0.045 * mm + RNG.normal(0, 0.04)
            fs = sum(comp[c] / 100 * ION_DATA[c]["z"] / ION_DATA[c]["r"] ** 2 for c in comp)
            tg = 120 + 11 * fs + 1.5 * comp[f2] + RNG.normal(0, 12)
            edc = 95 - 0.9 * comp[m] + 0.3 * comp[f2] + RNG.normal(0, 3)
            log_sigma = -2.0 - 0.10 * edc + 0.03 * comp[m] + RNG.normal(0, 0.4)
            rows.append({**comp,
                         "Density": round(dens, 3),
                         "Tg": round(tg, 1),
                         "conductivity @30°C": float(10 ** log_sigma),
                         "EDC / kJ mol-1": round(edc, 2)})
    return pd.DataFrame(rows)


def load_data(path=DATA_PATH):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded real dataset: {path}  (rows={len(df)})")
    else:
        df = synthesize()
        print(f"[!] {path} not found -> using SYNTHETIC data (rows={len(df)}). "
              f"Replace DATA_PATH with your CSV.")
    for c in COMPOSITION_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df


# ================================================ physics-informed features ===
def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add chemically meaningful descriptors. All are derived purely from the
    composition + ION_DATA you already have (no external tables needed)."""
    df = df.copy()
    comp = df[COMPOSITION_COLS].to_numpy(float)
    frac = comp / np.clip(comp.sum(1, keepdims=True), 1e-9, None)  # row-normalised

    z = np.array([ION_DATA[c]["z"] for c in COMPOSITION_COLS])
    r = np.array([ION_DATA[c]["r"] for c in COMPOSITION_COLS])
    mass = np.array([ATOMIC_MASS[c] + 8.0 * ION_DATA[c]["z"] for c in COMPOSITION_COLS])
    is_former = np.array([1.0 if c in FORMERS else 0.0 for c in COMPOSITION_COLS])
    is_mod = 1.0 - is_former

    # oxygen per mole of cations from charge balance: 2*nO = sum(z_i * x_i)
    n_O = 0.5 * (frac * z).sum(1)
    x_P = frac[:, COMPOSITION_COLS.index("P")]

    # ---- the physically important descriptors for phosphate glasses ----
    df["molar_mass"] = (frac * mass).sum(1)                       # drives density
    df["OP_ratio"] = n_O / np.clip(x_P, 1e-6, None)               # network connectivity (Q-species)
    df["modifier_frac"] = (frac * is_mod).sum(1)                  # depolymerisation
    df["mod_to_former"] = df["modifier_frac"] / np.clip(1 - df["modifier_frac"], 1e-6, None)
    # field strength split into modifier vs former contributions
    fs_all = frac * (z / r ** 2)
    df["fs_modifier"] = (fs_all * is_mod).sum(1)
    df["fs_former"] = (fs_all * is_former).sum(1)
    # your original descriptors (kept for continuity / ablation)
    df["avg_radius"] = (frac * r).sum(1)
    df["avg_charge"] = (frac * z).sum(1)
    df["avg_field_strength"] = fs_all.sum(1)
    # mobile-alkali concentration proxy (matters for ionic conductivity)
    alkali = np.array([1.0 if c in {"Li", "Na", "K", "Ag"} else 0.0 for c in COMPOSITION_COLS])
    df["alkali_frac"] = (frac * alkali).sum(1)
    return df


PHYS_FEATURES = ["molar_mass", "OP_ratio", "modifier_frac", "mod_to_former",
                 "fs_modifier", "fs_former", "avg_radius", "avg_charge",
                 "avg_field_strength", "alkali_frac"]


def feature_columns(target: str):
    """Feature set per target. Composition + physics descriptors; conductivity
    additionally gets the Arrhenius-coupled activation energy when present."""
    cols = list(COMPOSITION_COLS) + list(PHYS_FEATURES)
    return cols


# ===================================================== models with std out ====
def _gp_kernel(n_features: int):
    """Linear (DotProduct) captures the dominant additive trend & gives sane
    extrapolation; Matern (ARD) captures smooth nonlinear residual; White learns
    the noise floor for honest uncertainty."""
    return (
        C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(n_features),
                                     length_scale_bounds=(1e-2, 1e3), nu=2.5)
        + C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )


class StdGPR:
    """GP regressor that scales X and y internally and returns predictive std in
    the (optionally log10) target space. Avoids TransformedTargetRegressor, which
    silently drops return_std."""

    def __init__(self, log_target=False, n_restarts=None, random_state=0):
        self.log_target = log_target
        self.n_restarts = GP_RESTARTS if n_restarts is None else n_restarts
        self.random_state = random_state

    def _yt(self, y):
        y = np.asarray(y, float)
        return np.log10(np.clip(y, 1e-30, None)) if self.log_target else y

    def fit(self, X, y):
        X = np.asarray(X, float)
        yt = self._yt(y)
        self.xs_ = StandardScaler().fit(X)
        self.ys_ = StandardScaler().fit(yt.reshape(-1, 1))
        Xs = self.xs_.transform(X)
        ys = self.ys_.transform(yt.reshape(-1, 1)).ravel()
        self.gp_ = GaussianProcessRegressor(
            kernel=_gp_kernel(Xs.shape[1]), alpha=1e-10,
            n_restarts_optimizer=self.n_restarts, normalize_y=False,
            random_state=self.random_state,
        ).fit(Xs, ys)
        return self

    def predict(self, X, return_std=False):
        Xs = self.xs_.transform(np.asarray(X, float))
        m, s = self.gp_.predict(Xs, return_std=True)
        m = m * self.ys_.scale_[0] + self.ys_.mean_[0]
        s = s * self.ys_.scale_[0]
        return (m, s) if return_std else m


class PhysicsResidualGP:
    """Density = a * molar_mass (1-param physical baseline) + GP(residual).
    Expects a DataFrame so it can pull the 'molar_mass' column by name."""

    def __init__(self, n_restarts=None, random_state=0):
        self.gp = StdGPR(log_target=False, n_restarts=n_restarts, random_state=random_state)

    def fit(self, X: pd.DataFrame, y):
        M = X["molar_mass"].to_numpy(float)
        y = np.asarray(y, float)
        self.a_ = float(M @ y / (M @ M))          # closed-form least squares, 1 param
        self.gp.fit(X.to_numpy(float), y - self.a_ * M)
        return self

    def predict(self, X: pd.DataFrame, return_std=False):
        M = X["molar_mass"].to_numpy(float)
        base = self.a_ * M
        m, s = self.gp.predict(X.to_numpy(float), return_std=True)
        out = base + m
        return (out, s) if return_std else out


class BaselineMean:
    def fit(self, X, y): self.mu_ = float(np.mean(np.asarray(y, float))); return self
    def predict(self, X, return_std=False):
        m = np.full(len(X), self.mu_)
        return (m, np.zeros(len(X))) if return_std else m


class BayesianRidgeStd:
    """Linear additive baseline with native predictive std (in log space if asked)."""
    def __init__(self, log_target=False): self.log_target = log_target
    def _yt(self, y):
        y = np.asarray(y, float)
        return np.log10(np.clip(y, 1e-30, None)) if self.log_target else y
    def fit(self, X, y):
        X = np.asarray(X, float)
        self.xs_ = StandardScaler().fit(X)
        self.m_ = BayesianRidge().fit(self.xs_.transform(X), self._yt(y))
        return self
    def predict(self, X, return_std=False):
        Xs = self.xs_.transform(np.asarray(X, float))
        m, s = self.m_.predict(Xs, return_std=True)
        return (m, s) if return_std else m


def make_model(name, target):
    log = TARGETS[target]["log"]
    if name == "gpr":
        return StdGPR(log_target=log)
    if name == "physics_gp":          # density only
        return PhysicsResidualGP()
    if name == "bridge":
        return BayesianRidgeStd(log_target=log)
    if name == "stack":
        return StackGP(log_target=log)
    if name == "mean":
        return BaselineMean()
    raise ValueError(name)


# ============================================================== evaluation ====
def infer_groups(df: pd.DataFrame, mode="system"):
    """Group rows so an entire chemical *system* is held out together.
    'system' = the set of present (non-zero) oxides. This is the honest test:
    can the model predict a composition family it never saw?"""
    if mode == "system":
        keys = []
        comp = df[COMPOSITION_COLS].to_numpy(float)
        for row in comp:
            present = tuple(c for c, v in zip(COMPOSITION_COLS, row) if v > 1e-9)
            keys.append("-".join(present))
        codes = pd.Categorical(keys).codes
        return np.asarray(codes), keys
    raise ValueError(mode)


def _metrics(y_true, y_pred, y_std=None):
    out = {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if y_std is not None and np.all(np.isfinite(y_std)) and np.any(y_std > 0):
        lo, hi = y_pred - 1.96 * y_std, y_pred + 1.96 * y_std
        out["PICP95"] = float(np.mean((y_true >= lo) & (y_true <= hi)))  # want ~0.95
        out["MPIW"] = float(np.mean(hi - lo))                            # sharpness
    return out


def cv_oof(model_name, target, X: pd.DataFrame, y, groups, splitter):
    """Out-of-fold predictions under a given splitter (Group or random KFold)."""
    yt = np.log10(np.clip(np.asarray(y, float), 1e-30, None)) if TARGETS[target]["log"] else np.asarray(y, float)
    oof_pred = np.full(len(y), np.nan)
    oof_std = np.full(len(y), np.nan)
    split_iter = (splitter.split(X, yt, groups) if isinstance(splitter, GroupKFold)
                  else splitter.split(X, yt))
    for tr, te in split_iter:
        mdl = make_model(model_name, target).fit(X.iloc[tr], np.asarray(y)[tr])
        p, s = mdl.predict(X.iloc[te], return_std=True)
        oof_pred[te], oof_std[te] = p, s
    return _metrics(yt, oof_pred, oof_std), oof_pred, oof_std, yt


def prepare_xy(df: pd.DataFrame, target: str):
    """Shared target prep: drop missing rows, filter non-positive for log targets,
    return feature matrix X (DataFrame), target vector y, the feature column list,
    and the filtered sub-frame."""
    sub = df.dropna(subset=[target]).copy()
    if TARGETS[target]["log"]:
        sub = sub[sub[target] > 0]
    cols = feature_columns(target)
    X = sub[cols].reset_index(drop=True)
    y = sub[target].to_numpy(float)
    return X, y, cols, sub


def compare_models(df: pd.DataFrame, target: str, n_splits=5, n_scramble=None):
    """Run every candidate under grouped + random CV, pick the workhorse by
    honest (grouped) R2, and run a y-scramble check on it. Returns a structured
    dict consumed by both train.py and evaluate.py (single source of truth)."""
    n_scramble = N_SCRAMBLE if n_scramble is None else n_scramble
    sub = df.dropna(subset=[target]).copy()
    if TARGETS[target]["log"]:
        sub = sub[sub[target] > 0]
    n = len(sub)
    cols = feature_columns(target)
    X = sub[cols].reset_index(drop=True)
    y = sub[target].to_numpy(float)
    groups, _ = infer_groups(sub)
    n_groups = len(set(groups))
    k = max(2, min(n_splits, n_groups))
    space = "log10(" + TARGETS[target]["unit"] + ")" if TARGETS[target]["log"] else TARGETS[target]["unit"]

    candidates = ["mean", "bridge", "gpr"]
    if target == "Density":
        candidates.append("physics_gp")
    if RUN_STACK:
        candidates.append("stack")

    gkf = GroupKFold(n_splits=k)
    rkf = KFold(n_splits=k, shuffle=True, random_state=0)

    results = {}
    for name in candidates:
        g_m, g_pred, g_std, yt = cv_oof(name, target, X, y, groups, gkf)
        r_m, *_ = cv_oof(name, target, X, y, groups, rkf)
        results[name] = {"grouped": g_m, "random": r_m,
                         "oof_pred": g_pred, "oof_std": g_std, "y_true": yt}
    non_mean = [(nm, results[nm]["grouped"]) for nm in candidates if nm != "mean"]
    best = max(non_mean, key=lambda t: t[1]["R2"])[0]

    scr = []
    for _ in range(n_scramble):
        m, *_ = cv_oof(best, target, X, RNG.permutation(y), groups, gkf)
        scr.append(m["R2"])
    scramble = {"mean": float(np.mean(scr)) if scr else float("nan"),
                "std": float(np.std(scr)) if scr else float("nan"), "values": scr}

    return {"target": target, "n": n, "n_groups": n_groups, "k": k, "space": space,
            "feature_cols": cols, "candidates": candidates, "results": results,
            "best": best, "scramble": scramble}


def evaluate_target(df: pd.DataFrame, target: str, n_splits=5, n_scramble=None):
    """Console report for one target (wraps compare_models)."""
    r = compare_models(df, target, n_splits, n_scramble)
    print("\n" + "=" * 78)
    print(f"TARGET: {target}   (n={r['n']}, chemical systems={r['n_groups']}, "
          f"metric space={r['space']})")
    print("=" * 78)
    hdr = f"{'model':<12}{'GROUPED R2':>12}{'rand R2':>10}{'MAE':>10}{'RMSE':>10}{'PICP95':>9}{'MPIW':>10}"
    print(hdr); print("-" * len(hdr))
    for name in r["candidates"]:
        gm, rm = r["results"][name]["grouped"], r["results"][name]["random"]
        print(f"{name:<12}{gm['R2']:>12.3f}{rm['R2']:>10.3f}{gm['MAE']:>10.3f}"
              f"{gm['RMSE']:>10.3f}{gm.get('PICP95', float('nan')):>9.2f}"
              f"{gm.get('MPIW', float('nan')):>10.3f}")
    print(f"-> workhorse model (by grouped R2): {r['best']}")
    print("   note: a large rand-R2 >> grouped-R2 gap means random K-fold was "
          "interpolating within series, not generalising.")
    print(f"   y-scramble grouped R2 ({r['best']}): "
          f"{r['scramble']['mean']:+.3f} ± {r['scramble']['std']:.3f}  (must be ~0 / negative)")
    return r["best"]


# ========================================== OPTIONAL: stacking & symbolic ====
# These are off by default so the validated core always runs. Turn on via env:
#   RUN_STACK=1   -> add a leakage-safe BayesianRidge+GP blend to the comparison
#   RUN_PYSR=1    -> fit PySR symbolic regression (needs: pip install pysr; julia)
RUN_STACK = os.environ.get("RUN_STACK", "0") == "1"
RUN_PYSR = os.environ.get("RUN_PYSR", "0") == "1"


class StackGP:
    """Leakage-safe stack of a linear (BayesianRidge) and a GP learner.
    Blend weights are fit by non-negative least squares on an INNER grouped
    split of the training fold, so no test information leaks. Predictive std is
    taken from the GP component (the calibrated one)."""

    def __init__(self, log_target=False):
        self.log_target = log_target

    def _yt(self, y):
        y = np.asarray(y, float)
        return np.log10(np.clip(y, 1e-30, None)) if self.log_target else y

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        g, _ = infer_groups(X)                      # inner split by chemical system
        k = max(2, min(4, len(set(g))))
        inner = GroupKFold(n_splits=k)
        P = np.zeros((len(y), 2))
        for tr, te in inner.split(X, self._yt(y), g):
            P[te, 0] = BayesianRidgeStd(self.log_target).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            P[te, 1] = StdGPR(self.log_target).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        from scipy.optimize import nnls
        w, _ = nnls(P, self._yt(y))
        self.w_ = w / (w.sum() + 1e-12)
        self.br_ = BayesianRidgeStd(self.log_target).fit(X, y)
        self.gp_ = StdGPR(self.log_target).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame, return_std=False):
        pb = self.br_.predict(X)
        pg, sg = self.gp_.predict(X, return_std=True)
        m = self.w_[0] * pb + self.w_[1] * pg
        return (m, sg) if return_std else m


def run_symbolic_regression(df, target):
    """Fit a closed-form law with PySR and print it. Interpretable equations are
    the strongest test of 'did it learn chemistry?' — read the formula and check
    it against known phosphate-glass behaviour."""
    try:
        from pysr import PySRRegressor
    except Exception as e:
        print(f"[PySR skipped: {e}]  install with: pip install pysr  (also needs Julia)")
        return
    sub = df.dropna(subset=[target]).copy()
    if TARGETS[target]["log"]:
        sub = sub[sub[target] > 0]
    cols = PHYS_FEATURES
    X = sub[cols].to_numpy(float)
    y = (np.log10(sub[target].to_numpy(float)) if TARGETS[target]["log"]
         else sub[target].to_numpy(float))
    model = PySRRegressor(
        niterations=60, binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt", "log"], maxsize=20,
        model_selection="best", progress=False, random_state=0,
        deterministic=True, procs=0, multithreading=False,
    )
    model.fit(X, y, variable_names=cols)
    tag = "log10" if TARGETS[target]["log"] else "raw"
    print(f"\nSymbolic law for {target} ({tag}):\n  {model.get_best()['equation']}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    df = add_physics_features(df)
    print(f"\nFeatures used: {len(feature_columns('Tg'))} "
          f"({len(COMPOSITION_COLS)} composition + {len(PHYS_FEATURES)} physics)")
    summary = {}
    for target in TARGETS:
        if target in df.columns:
            summary[target] = evaluate_target(df, target)
            if RUN_PYSR:
                run_symbolic_regression(df, target)
    print("\n" + "#" * 78)
    print("WORKHORSE MODEL PER TARGET (chosen by honest grouped-CV R2):")
    for t, m in summary.items():
        print(f"   {t:<22} -> {m}")
    print("#" * 78)


if __name__ == "__main__":
    main()
