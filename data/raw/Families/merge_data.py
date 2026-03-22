import pandas as pd
from pathlib import Path

def aggregate_glass_data(root_folder_path: str, output_filename: str = "global_glass_dataset.csv"):
   
    root_dir = Path(root_folder_path)
    if not root_dir.exists() or not root_dir.is_dir():
        return None

    all_dataframes = []
    
    files_to_process = list(root_dir.rglob("*.csv")) + list(root_dir.rglob("*.xlsx"))
    
    if not files_to_process:
        return None
        

    for file_path in files_to_process:
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path)
            
            if df.empty:
                continue
                
            cols_to_drop = [col for col in df.columns if 'Unnamed' in str(col)]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            df['source_file'] = file_path.name
            all_dataframes.append(df)
            
        except Exception as e:
            print(f"[ERROR] Greška pri učitavanju {file_path.name}: {e}")

    if not all_dataframes:
        return None


    global_df = pd.concat(all_dataframes, ignore_index=True)
    
    targets = ['Density', 'Tg', 'conductivity @30°C', 'EDC / kJ mol-1']
    existing_targets = [col for col in targets if col in global_df.columns]
    
    meta_cols = ['source_file', 'reference:', 'mol%']
    existing_meta = [col for col in meta_cols if col in global_df.columns]
    
    elements = [col for col in global_df.columns if col not in existing_targets and col not in existing_meta]
    
    for col in existing_targets + elements:
        global_df[col] = pd.to_numeric(global_df[col], errors='coerce')
    
    global_df[elements] = global_df[elements].fillna(0)
    
    elements.sort()
    
    if 'P' in elements:
        elements.remove('P')
        elements.insert(0, 'P')
        
    new_column_order = elements + existing_targets + existing_meta
    global_df = global_df[new_column_order]
    
    global_df.to_csv(output_filename, index=False)
    
    return global_df

if __name__ == "__main__":
    PROJECT_ROOT = "." 
    df = aggregate_glass_data(root_folder_path=PROJECT_ROOT)