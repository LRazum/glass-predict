"""
merge_data_v3.py
================
Drop-in replacement for data/raw/Families/merge_data.py.

The only substantive change vs. your original: the merge now PRESERVES the
series identity. Your original stored `file_path.name`, and because every
family folder contains a file called `data.xlsx`, every merged row ended up
labelled identically — the information that tells us which experimental
series a glass belongs to was thrown away at merge time.

This version writes two extra columns:
    series_id : relative path of the source file, unique per series file
                (e.g. "AgPO3_WO3/data.xlsx")
    family    : the immediate parent folder name

glass_pipeline_v3.assign_groups(grouping="series") picks up `series_id`
automatically — no other change needed downstream. If you re-run
scripts/feature_engineering.py on the merged file, note that its `elements`
list is built by exclusion, so 'series_id' and 'family' are added to
meta_cols here and must be excluded there too (v3 builds its own features
internally, so the simplest path is to point v3 at this file directly).
"""
import pandas as pd
from pathlib import Path


def aggregate_glass_data(root_folder_path: str,
                         output_filename: str = "global_glass_dataset.csv"):
    root_dir = Path(root_folder_path)
    if not root_dir.exists() or not root_dir.is_dir():
        return None

    all_dataframes = []
    files_to_process = (list(root_dir.rglob("*.csv"))
                        + list(root_dir.rglob("*.xlsx")))
    files_to_process = [f for f in files_to_process
                        if f.name != output_filename]
    if not files_to_process:
        return None

    for file_path in files_to_process:
        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            if df.empty:
                continue

            cols_to_drop = [c for c in df.columns if "Unnamed" in str(c)]
            df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

            df["source_file"] = file_path.name
            # >>> the fix: keep the folder identity = the series identity <<<
            df["series_id"] = str(file_path.relative_to(root_dir))
            df["family"] = file_path.parent.name
            all_dataframes.append(df)
        except Exception as e:
            print(f"[ERROR] Greška pri učitavanju {file_path.name}: {e}")

    if not all_dataframes:
        return None

    global_df = pd.concat(all_dataframes, ignore_index=True)

    targets = ["Density", "Tg", "conductivity @30°C", "EDC / kJ mol-1"]
    existing_targets = [c for c in targets if c in global_df.columns]

    meta_cols = ["source_file", "series_id", "family", "reference:", "mol%"]
    existing_meta = [c for c in meta_cols if c in global_df.columns]

    elements = [c for c in global_df.columns
                if c not in existing_targets and c not in existing_meta]

    for col in existing_targets + elements:
        global_df[col] = pd.to_numeric(global_df[col], errors="coerce")
    global_df[elements] = global_df[elements].fillna(0)

    elements.sort()
    if "P" in elements:
        elements.remove("P")
        elements.insert(0, "P")

    global_df = global_df[elements + existing_targets + existing_meta]
    global_df.to_csv(output_filename, index=False)

    n_series = global_df["series_id"].nunique()
    print(f"Merged {len(global_df)} rows from {n_series} series files "
          f"-> {output_filename}")
    return global_df


if __name__ == "__main__":
    aggregate_glass_data(root_folder_path=".")