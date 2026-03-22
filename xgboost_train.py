import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

def create_parity_plot(y_train, y_pred_train, y_test, y_pred_test, target_name, apply_log10, filepath):
    """Creates and saves a Parity Plot comparing actual and predicted properties."""
    plt.figure(figsize=(7, 6))
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    plt.scatter(y_train, y_pred_train, alpha=0.5, label=f'Train (R²={r2_train:.2f})')
    plt.scatter(y_test, y_pred_test, alpha=0.8, color='red', label=f'Test (R²={r2_test:.2f})')
    
    min_val = min(y_train.min(), y_test.min(), y_pred_train.min(), y_pred_test.min())
    max_val = max(y_train.max(), y_test.max(), y_pred_train.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect (y=x)')
    
    target_display = f"log10({target_name})" if apply_log10 else target_name
    plt.xlabel(f"Actual measured: {target_display}")
    plt.ylabel(f"Model predicts: {target_display}")
    plt.title(f"Parity Plot: {target_name}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(filepath, dpi=300)
    plt.close()

def optimize_and_train(df, target_name, features, apply_log10=False, output_dir="global_models"):
    """Performs optimization, trains the model, and saves the results."""
    
    df_clean = df.dropna(subset=[target_name]).copy()
    
    if apply_log10:
        df_clean = df_clean[df_clean[target_name] > 0]
        y = np.log10(df_clean[target_name])
    else:
        y = df_clean[target_name]
        
    X = df_clean[features]
    n_samples = len(X)
    
    if n_samples < 20:
        print(f"[SKIPPED] {target_name} - Insufficient data (n={n_samples}).")
        return

    print(f"Processing: {target_name} (n={n_samples})...", end=" ", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_pipeline = make_pipeline(
        StandardScaler(), 
        XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42)
    )
    
    param_grid = {
        'xgbregressor__n_estimators': [50, 100],
        'xgbregressor__max_depth': [2, 3],
        'xgbregressor__learning_rate': [0.05, 0.1],
        'xgbregressor__reg_lambda': [10.0, 50.0],
        'xgbregressor__min_child_weight': [5, 10]
    }
    
    grid_search = GridSearchCV(
        xgb_pipeline, 
        param_grid, 
        cv=5, 
        scoring='r2', 
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluation
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # Saving files
    os.makedirs(output_dir, exist_ok=True)
    safe_name = target_name.replace(" ", "_").replace("/", "_").replace("°", "")
    
    model_path = os.path.join(output_dir, f"model_xgb_{safe_name}.joblib")
    plot_path = os.path.join(output_dir, f"parity_xgb_{safe_name}.png")
    
    joblib.dump({
        'model': best_model.named_steps['xgbregressor'], 
        'scaler': best_model.named_steps['standardscaler'], 
        'features': features
    }, model_path)
    
    create_parity_plot(y_train, y_pred_train, y_test, y_pred_test, target_name, apply_log10, plot_path)
    
    # Final output
    print(f"Done! | Train R²: {r2_train:.3f} | Test R²: {r2_test:.3f} | MAE: {mae_test:.3f}")

def main():
    input_file = os.path.join("data", "processed", "global_dataset_with_features.csv")

    if not os.path.exists(input_file):
        print(f"[ERROR] Cannot find file: {input_file}")
        return
        
    df = pd.read_csv(input_file)
    targets = ['Density', 'Tg', 'conductivity @30°C', 'EDC / kJ mol-1']
    meta_cols = ['source_file', 'reference:', 'mol%']
    features = [col for col in df.columns if col not in targets and col not in meta_cols]
    
    print("--- Starting XGBoost model training ---")
    for target in targets:
        if target in df.columns:
            needs_log = (target == 'conductivity @30°C')
            optimize_and_train(df, target, features, apply_log10=needs_log)

if __name__ == "__main__":
    main()