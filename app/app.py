import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Physical constants of ions needed for descriptor calculation
ION_DATA = {
    "Li": {"r": 0.76, "z": 1}, "Na": {"r": 1.02, "z": 1}, "K": {"r": 1.38, "z": 1},
    "Mg": {"r": 0.72, "z": 2}, "Ca": {"r": 1.00, "z": 2}, "Sr": {"r": 1.18, "z": 2}, "Ba": {"r": 1.35, "z": 2},
    "Zn": {"r": 0.74, "z": 2}, "Pb": {"r": 1.19, "z": 2}, "Ag": {"r": 1.15, "z": 1},
    "Ti": {"r": 0.61, "z": 4}, "Nb": {"r": 0.64, "z": 5}, "V":  {"r": 0.54, "z": 5},
    "Mo": {"r": 0.59, "z": 6}, "W":  {"r": 0.60, "z": 6}, "Fe": {"r": 0.64, "z": 3},
    "Al": {"r": 0.53, "z": 3}, "B":  {"r": 0.23, "z": 3}, "P":  {"r": 0.38, "z": 5},
    "Ge": {"r": 0.53, "z": 4}, "Te": {"r": 0.97, "z": 4}, "Ce": {"r": 1.01, "z": 3},
    "Bi": {"r": 1.03, "z": 3}, "Sb": {"r": 0.76, "z": 3}
}

APP_DIR = Path(__file__).parent

def calculate_descriptors(composition):
    """Calculates the normalized composition and physical descriptors."""
    total_mol = sum(composition.values())
    if total_mol == 0:
        return None, None
        
    norm_comp = {k: (v / total_mol) * 100 for k, v in composition.items()}
    sum_r, sum_z, sum_fs = 0.0, 0.0, 0.0
    
    for el, mol in norm_comp.items():
        if el in ION_DATA:
            r = ION_DATA[el]["r"]
            z = ION_DATA[el]["z"]
            frac = mol / 100.0
            sum_r += frac * r
            sum_z += frac * z
            sum_fs += frac * (z / (r**2))
            
    descriptors = {
        **norm_comp,
        'avg_radius': sum_r,
        'avg_charge': sum_z,
        'avg_field_strength': sum_fs
    }
    return norm_comp, descriptors

def get_available_targets(models_dir=APP_DIR):
    """Scans the directory and finds for which properties trained models exist."""
    path = Path(models_dir)
    if not path.exists():
        return []
    targets = sorted(list(set([f.stem.replace("model_gpr_", "") for f in path.glob("model_gpr_*.joblib")])))
    return targets

def main():
    st.set_page_config(page_title="Phosphate Glass Predictor", layout="wide")
    
    st.title("Phosphate Glass Property Predictor")
    
    # Adding the disclaimer from the README document
    st.markdown("⚠️ **Project Status: Early Prototype** *This project is currently in its initial development phase. It is a working version intended primarily for testing, research, and validation purposes, not a final production-ready tool.*")
    
    st.markdown("Enter the molar composition of the glass. The system will automatically normalize the fractions to 100% and apply XGBoost and GPR models for property prediction.")
    st.markdown("---")

    st.sidebar.header("Composition (mol %)")
    st.sidebar.markdown("Enter raw values. The system performs normalization.")
    
    elements = ["P", "Na", "Fe", "V", "Li", "Ag", "W", "Zn", "Al", "K", "Mg", "Ca", "Ba", "Pb", "B"]
    user_comp = {}
    
    for el in elements:
        default_val = 50.0 if el == "P" else 0.0
        user_comp[el] = st.sidebar.number_input(f"{el} (mol%)", min_value=0.0, max_value=100.0, value=default_val, step=1.0)

    total_input = sum(user_comp.values())
    
    if total_input == 0:
        st.warning("Please enter values greater than zero for at least one element.")
        return

    norm_comp, descriptors = calculate_descriptors(user_comp)
    
    with st.expander("Overview of normalized composition and descriptors", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Normalized fractions (actual model inputs):**")
            df_norm = pd.DataFrame([norm_comp]).T
            df_norm.columns = ["mol %"]
            st.dataframe(df_norm[df_norm["mol %"] > 0].style.format("{:.2f}"))
        with c2:
            st.markdown("**Calculated physical descriptors:**")
            st.write(f"- Average radius: {descriptors['avg_radius']:.4f} Å")
            st.write(f"- Average charge: {descriptors['avg_charge']:.4f}")
            st.write(f"- Field Strength: {descriptors['avg_field_strength']:.4f}")

    targets = get_available_targets()
    
    if not targets:
        st.error("No models found in the directory. Please ensure the training scripts were executed successfully.")
        return

    st.header("Prediction Results")
    
    cols = st.columns(2)
    
    for idx, target in enumerate(targets):
        col = cols[idx % 2]
        with col:
            st.subheader(target.replace('_', ' ').title())
            
            xgb_path = APP_DIR / f"model_xgb_{target}.joblib"
            xgb_pred = None
            if xgb_path.exists():
                try:
                    xgb_data = joblib.load(xgb_path)
                    df_in = pd.DataFrame([{f: descriptors.get(f, 0.0) for f in xgb_data['features']}])
                    X_sc = xgb_data['scaler'].transform(df_in)
                    xgb_pred = xgb_data['model'].predict(X_sc)[0]
                except Exception as e:
                    st.error(f"XGBoost error: {e}")

            gpr_path = APP_DIR / f"model_gpr_{target}.joblib"
            gpr_pred, gpr_err = None, None
            if gpr_path.exists():
                try:
                    gpr_data = joblib.load(gpr_path)
                    df_in = pd.DataFrame([{f: descriptors.get(f, 0.0) for f in gpr_data['features']}])
                    X_sc = gpr_data['x_scaler'].transform(df_in)
                    p_sc, s_sc = gpr_data['model'].predict(X_sc, return_std=True)
                    
                    gpr_pred = gpr_data['y_scaler'].inverse_transform(p_sc.reshape(-1, 1)).ravel()[0]
                    gpr_err = s_sc[0] * gpr_data['y_scaler'].scale_[0] * 1.96
                except Exception as e:
                    st.error(f"GPR error: {e}")

            is_log = "conductivity" in target.lower()
            unit = " S/cm" if is_log else ""
            prefix = "10^" if is_log else ""

            if xgb_pred is not None:
                st.markdown(f"**XGBoost:** {prefix}{xgb_pred:.2f}{unit}")
            
            if gpr_pred is not None and gpr_err is not None:
                st.markdown(f"**GPR (Bayes):** {prefix}{gpr_pred:.2f} ± {gpr_err:.2f}{unit}")
            
            st.markdown("---")

if __name__ == "__main__":
    main()