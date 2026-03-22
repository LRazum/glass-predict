import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.exceptions import ConvergenceWarning

# Disabling convergence warnings due to intentional kernel restrictions
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

def train_regularized_gpr(df, target_name, features, apply_log10=False, output_dir="global_models"):
    """
    Trains, optimizes, and saves a Gaussian Process Regressor for the selected target variable.
    Includes scaling of input and output variables and kernel restrictions to prevent overfitting.
    """
    df_clean = df.dropna(subset=[target_name]).copy()
    
    if apply_log10:
        df_clean = df_clean[df_clean[target_name] > 0]
        y = np.log10(df_clean[target_name])
    else:
        y = df_clean[target_name]
        
    X = df_clean[features]
    n_samples = len(X)
    
    if n_samples < 20:
        print(f"Skipping '{target_name}': insufficient number of samples (n={n_samples}).\n")
        return

    print(f"--- Training GPR model for: {target_name} (n={n_samples}) ---")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling input variables (X)
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # Scaling output variables (y) for stability of GPR optimization
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # Restricting kernel variance and Matern length (length_scale)
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-1, 1e1)) * \
             Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e2), nu=1.5)
    
    gpr = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=15,
        random_state=42
    )

    # Searching for the optimal regularization level
    param_grid = {
        'alpha': [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    }

    grid_search = GridSearchCV(
        gpr, 
        param_grid, 
        cv=5, 
        scoring='r2', 
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train_scaled)
    best_gpr = grid_search.best_estimator_
    
    print("Optimized parameters:")
    print(f"  - Alpha:  {grid_search.best_params_['alpha']}")
    print(f"  - Kernel: {best_gpr.kernel_}")

    # Prediction in scaled space
    y_pred_train_scaled, std_train_scaled = best_gpr.predict(X_train_scaled, return_std=True)
    y_pred_test_scaled, std_test_scaled = best_gpr.predict(X_test_scaled, return_std=True)
    
    # Inverse transformation of results to original units
    y_pred_train = y_scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
    y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
    
    # Adjusting standard deviation by the scaling factor
    std_test = std_test_scaled * y_scaler.scale_[0]

    # Model performance evaluation
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print("Evaluation results:")
    print(f"  - Train R²: {r2_train:.4f}")
    print(f"  - Test R²:  {r2_test:.4f}")
    print(f"  - Test MAE: {mae_test:.4f}")
    
    # Saving GPR model and scaler
    os.makedirs(output_dir, exist_ok=True)
    safe_name = target_name.replace(" ", "_").replace("/", "_").replace("°", "")
    
    model_filepath = os.path.join(output_dir, f"model_gpr_{safe_name}.joblib")
    joblib.dump({
        'model': best_gpr, 
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'features': features
    }, model_filepath)
    print(f"Model saved: {model_filepath}")
    
    # Generating Parity Plot with 95% confidence interval
    plot_path = os.path.join(output_dir, f"parity_gpr_{safe_name}.png")
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_train, y_pred_train, alpha=0.3, label=f'Train (R²={r2_train:.2f})', color='blue')
    plt.errorbar(y_test, y_pred_test, yerr=std_test*1.96, fmt='o', color='red', 
                 ecolor='lightcoral', elinewidth=2, capsize=4, alpha=0.8, 
                 label=f'Test (R²={r2_test:.2f}) with 95% CI')
    
    min_val = min(y_train.min(), y_test.min(), y_pred_train.min(), y_pred_test.min())
    max_val = max(y_train.max(), y_test.max(), y_pred_train.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal (y=x)')
    
    target_display = f"log10({target_name})" if apply_log10 else target_name
    plt.xlabel(f"Actual value: {target_display}")
    plt.ylabel(f"Predicted value (GPR): {target_display}")
    plt.title(f"Gaussian Process Regression: {target_name}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved:  {plot_path}\n")

def main():
    input_file = os.path.join("data", "processed", "global_dataset_with_features.csv")
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return
        
    df = pd.read_csv(input_file)
    targets = ['Density', 'Tg', 'conductivity @30°C', 'EDC / kJ mol-1']
    meta_cols = ['source_file', 'reference:', 'mol%']
    features = [col for col in df.columns if col not in targets and col not in meta_cols]
    
    for target in targets:
        if target in df.columns:
            needs_log = (target == 'conductivity @30°C')
            train_regularized_gpr(df, target, features, apply_log10=needs_log)

if __name__ == "__main__":
    main()