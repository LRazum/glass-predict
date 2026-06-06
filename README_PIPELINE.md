# Phosphate-Glass Property Pipeline (modelling core)

Predicts **Density, Tg, DC conductivity, and DC activation energy** of phosphate
glasses from molar composition, with calibrated uncertainty. This is the
ML/evaluation core only — the web app comes later.

## Files

| file | role |
|---|---|
| `glass_pipeline_v2.py` | **library**: feature engineering, models, grouped-CV harness, model selection |
| `train.py` | fit the winning model per target on all data → `models/*.joblib` + `manifest.json` |
| `evaluate.py` | honest metrics report + parity/calibration plots → `reports/` |
| `predict.py` | load a model and predict a new composition (with 95% interval) |
| `requirements.txt` | dependencies |

Keep these four `.py` files in the same folder (they import `glass_pipeline_v2`).

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
# 1) evaluate — see the honest (leave-one-system-out) numbers and figures
python evaluate.py

# 2) train — persist the chosen model per target
python train.py

# 3) predict — query a new glass
python predict.py --comp "P=50,Na=35,Al=15"
python predict.py --csv my_new_glasses.csv          # one composition per row
```

## Point it at your real data

By default the scripts look for the dataset at
`data/processed/global_dataset_with_features.csv`. **If that file is missing they
fall back to synthetic data so the code runs out of the box — those numbers are
illustrative only.** To use your real dataset, set `GLASS_DATA` (the scripts
recompute all physics features from the raw composition, so the raw CSV is fine):

```bash
GLASS_DATA=data/raw/global_glass_dataset.csv python evaluate.py
GLASS_DATA=data/raw/global_glass_dataset.csv GP_RESTARTS=8 python train.py
```

Your CSV must contain the composition columns
(`P, Ag, Al, Fe, K, Li, Mo, Na, Nb, V, W, Zn`, in mol%) and the target columns
(`Density, Tg, conductivity @30°C, EDC / kJ mol-1`). Missing target cells are fine —
each target is trained on whatever rows have it.

## What to look for in the evaluation

- **`grouped R²` vs `random R²`** — grouped is the honest number (an entire
  chemical system held out). A large gap means random K-fold was just
  interpolating within a composition series.
- **`PICP95` ≈ 0.95** — the 95% interval really covers ~95% of points.
- **`MPIW`** — interval width (sharpness); lower is better at equal coverage.
- **y-scramble grouped R² ≈ 0** — confirms the model learns chemistry, not noise.
- **calibration plot** — points should sit on the diagonal.

## Environment knobs

| var | default | effect |
|---|---|---|
| `GLASS_DATA` | processed CSV path | dataset to use |
| `GP_RESTARTS` | 3 | GP optimiser restarts (0–1 fast iteration, 8+ for final numbers) |
| `N_SCRAMBLE` | 3 | y-scramble repeats |
| `MODELS_DIR` | `models` | where models are saved/loaded |
| `REPORTS_DIR` | `reports` | where the report/figures go |
| `RUN_STACK` | 0 | add the BayesianRidge+GP stacking blend to the comparison |
| `RUN_LC` | 0 | also draw learning curves in `evaluate.py` |
| `RUN_PYSR` | 0 | fit PySR symbolic regression (needs Julia; see requirements) |

## Notes

- Conductivity is modelled in `log10` space; `predict.py` reports it back in S/cm
  with an asymmetric 95% interval.
- Density uses a 1-parameter physical baseline (density ∝ molar mass) plus a GP on
  the residual; the other targets use a Linear+Matérn(ARD)+White GP, with
  BayesianRidge as a strong baseline. The winner per target is chosen by grouped R².
- For best grouping, add a `series_id`/`reference` column to your data and group on
  that instead of the "set of present oxides" heuristic in `infer_groups()`.
