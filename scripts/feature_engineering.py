import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ION_DATA = {
    "Li": {"r": 0.76, "z": 1}, "Na": {"r": 1.02, "z": 1}, "K": {"r": 1.38, "z": 1},
    "Mg": {"r": 0.72, "z": 2}, "Ca": {"r": 1.00, "z": 2}, "Sr": {"r": 1.18, "z": 2},
    "Zn": {"r": 0.74, "z": 2}, "Pb": {"r": 1.19, "z": 2}, "Ag": {"r": 1.15, "z": 1},
    "Ti": {"r": 0.61, "z": 4}, "Nb": {"r": 0.64, "z": 5}, "V":  {"r": 0.54, "z": 5},
    "Mo": {"r": 0.59, "z": 6}, "W":  {"r": 0.60, "z": 6}, "Fe": {"r": 0.64, "z": 3},
    "Al": {"r": 0.53, "z": 3}, "B":  {"r": 0.23, "z": 3}, "P":  {"r": 0.38, "z": 5},
    "Ge": {"r": 0.53, "z": 4}, "Te": {"r": 0.97, "z": 4}, "Ce": {"r": 1.01, "z": 3},
    "Bi": {"r": 1.03, "z": 3}, "Sb": {"r": 0.76, "z": 3}, "Ba": {"r": 1.35, "z": 2}
}

def calculate_physical_features(row, elements):
    total_mol = 0.0
    sum_r = 0.0
    sum_z = 0.0
    sum_fs = 0.0
    
    for el in elements:
        mol_percent = row.get(el, 0.0)
        if pd.isna(mol_percent) or mol_percent <= 0:
            continue
            
        if el in ION_DATA:
            r = ION_DATA[el]["r"]
            z = ION_DATA[el]["z"]
            fs = z / (r ** 2)
            
            total_mol += mol_percent
            sum_r += mol_percent * r
            sum_z += mol_percent * z
            sum_fs += mol_percent * fs
            
    if total_mol > 0:
        return pd.Series({
            'avg_radius': sum_r / total_mol,
            'avg_charge': sum_z / total_mol,
            'avg_field_strength': sum_fs / total_mol
        })
    else:
        return pd.Series({'avg_radius': np.nan, 'avg_charge': np.nan, 'avg_field_strength': np.nan})

def main(input_csv="global_glass_dataset.csv", output_csv="global_dataset_with_features.csv"):
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        return
        
    targets = ['Density', 'Tg', 'conductivity @30°C', 'EDC / kJ mol-1']
    meta_cols = ['source_file', 'reference:', 'mol%']
    
    elements = [col for col in df.columns if col not in targets and col not in meta_cols]
    
    features_df = df.apply(calculate_physical_features, axis=1, elements=elements)
    df_enriched = pd.concat([df, features_df], axis=1)
    
    df_enriched.to_csv(output_csv, index=False)
    
    
    cols_for_corr = list(features_df.columns) + targets
    
    corr_matrix = df_enriched[cols_for_corr].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation matrix')
    plt.tight_layout()
    
    plot_filename = "correlation_heatmap.png"
    plt.savefig(plot_filename, dpi=300)
    
if __name__ == "__main__":
    main()