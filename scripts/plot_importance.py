import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import os
from sklearn.inspection import permutation_importance
import warnings

# Ignore warnings for a cleaner console output
warnings.filterwarnings("ignore")

# Mapping the "safe" filename strings back to the original target column names in the CSV
TARGET_MAPPING = {
    'Density': 'Density',
    'Tg': 'Tg',
    'conductivity_@30C': 'conductivity @30°C',
    'EDC___kJ_mol-1': 'EDC / kJ mol-1'
}

def plot_feature_importances(models_dir="app", data_path="data/processed/global_dataset_with_features.csv"):
    """
    Loads saved models and generates feature importance plots.
    - For XGBoost: Uses built-in 'feature_importances_'.
    - For GPR: Uses 'Permutation Importance' (requires the dataset).
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Error: Directory '{models_dir}' does not exist.")
        return

    print(f"Analyzing models in '{models_dir}' and generating importance plots...\n")

    # Load dataset if available (required for GPR permutation importance)
    df = None
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Fallback to the root directory just in case
        alt_path = "global_dataset_with_features.csv"
        if os.path.exists(alt_path):
            df = pd.read_csv(alt_path)
        else:
            print(f"Warning: Dataset not found at '{data_path}'.")
            print("GPR models will be skipped because they require data for Permutation Importance.\n")

    # Loop through all saved models
    for model_file in models_path.glob("*.joblib"):
        try:
            payload = joblib.load(model_file)
            model = payload.get('model')
            features = payload.get('features')

            if model is None or features is None:
                continue

            is_xgb = "xgb" in model_file.name.lower()
            is_gpr = "gpr" in model_file.name.lower()

            # Extract the target name from the filename
            target_safe_name = model_file.stem.replace("model_xgb_", "").replace("model_gpr_", "")
            target_display_name = target_safe_name.replace("_", " ")

            importances = None

            # 1. XGBoost Feature Importance
            if is_xgb and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                xlabel = 'Relative Importance'
                title_prefix = 'Feature Importance (XGBoost)'

            # 2. GPR Permutation Importance
            elif is_gpr:
                if df is None:
                    continue  # Skip GPR if the dataset wasn't found

                target_col = TARGET_MAPPING.get(target_safe_name)
                if not target_col or target_col not in df.columns:
                    print(f" -> Skipping {model_file.name}: Target column not found in dataset.")
                    continue

                # Prepare the data exactly as it was prepared during training
                df_clean = df.dropna(subset=[target_col]).copy()
                if target_col == 'conductivity @30°C':
                    df_clean = df_clean[df_clean[target_col] > 0]
                    y = np.log10(df_clean[target_col])
                else:
                    y = df_clean[target_col]

                X = df_clean[features]
                x_scaler = payload.get('x_scaler')
                y_scaler = payload.get('y_scaler')

                if not x_scaler or not y_scaler:
                    print(f" -> Skipping {model_file.name}: Missing scalers inside the joblib file.")
                    continue

                # Scale the inputs and outputs
                X_scaled = x_scaler.transform(X)
                y_scaled = y_scaler.transform(y.values.reshape(-1, 1)).ravel()

                print(f" -> Calculating Permutation Importance for {model_file.name} (this may take a few seconds)...")
                # Perform the permutation algorithm
                result = permutation_importance(model, X_scaled, y_scaled, n_repeats=10, random_state=42, n_jobs=-1)
                importances = result.importances_mean
                xlabel = 'Permutation Importance (Mean Decrease in R²)'
                title_prefix = 'Feature Importance (GPR)'

            # 3. Plotting the results
            if importances is not None:
                feat_imp = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True)

                # Clip negative values to 0 (Permutation importance can sometimes produce tiny negative noise for useless features)
                feat_imp['Importance'] = feat_imp['Importance'].clip(lower=0)

                top_10 = feat_imp.tail(10)

                plt.figure(figsize=(8, 6))
                plt.barh(top_10['Feature'], top_10['Importance'], color='steelblue', edgecolor='black')

                plt.xlabel(xlabel)
                plt.ylabel('Feature')
                plt.title(f'{title_prefix}: {target_display_name}')
                plt.grid(axis='x', linestyle=':', alpha=0.7)
                plt.tight_layout()

                # Save the plot
                out_file = models_path / f"importance_{model_file.stem}.png"
                plt.savefig(out_file, dpi=300)
                plt.close()
                print(f" -> Generated plot: {out_file.name}")

        except Exception as e:
            print(f" -> Error processing {model_file.name}: {e}")

if __name__ == "__main__":
    # Define directories matching your VS Code structure
    plot_importance_dir = "app"
    dataset_location = "data/processed/global_dataset_with_features.csv"
    
    plot_feature_importances(models_dir=plot_importance_dir, data_path=dataset_location)