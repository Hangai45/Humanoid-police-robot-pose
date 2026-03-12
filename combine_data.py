import os
import pandas as pd
import glob

# =========================
# PATHS
# =========================
BASE_DIR = r"C:\Users\DELL.DESKTOP-PTQ10MO\Desktop\WEB\police robot\version_2"
FEATURES_DIR = os.path.join(BASE_DIR, "dataset", "features")
MERGED_FILE = os.path.join(BASE_DIR, "dataset", "merged_features.csv")

import re

# Стандарт баганын дараалал (18 angles + 4 vertical diffs + 4 meta = 22)
STD_COLS = [
    "timestamp", "label", "burst_id", "frame_idx", 
    "LS", "RS", "LE", "RE", 
    "RH_T", "RH_I", "RH_M", "RH_R", "RH_P", 
    "LH_T", "LH_I", "LH_M", "LH_R", "LH_P",
    "RW_Y", "LW_Y", "RE_Y", "LE_Y"
]

def normalize_columns(df):
    """
    Баганын нэрсийг стандарт болгох. 
    Жишээ нь: 'Зүүн мөрний өнцөг (LS)' -> 'LS'
    """
    new_cols = {}
    for col in df.columns:
        # Хаалтан доторх кодыг хайх (LS, RS гэх мэт)
        match = re.search(r'\(([^)]+)\)', col)
        if match:
            new_cols[col] = match.group(1)
        else:
            new_cols[col] = col
    
    df = df.rename(columns=new_cols)
    
    # Бүх стандарт баганууд байгаа эсэхийг шалгах, байхгүй бол 0-ээр дүүргэх
    for c in STD_COLS:
        if c not in df.columns:
            df[c] = 0.0
            
    return df[STD_COLS]

def combine_data():
    print(f"Searching for files in: {FEATURES_DIR} and {os.path.dirname(FEATURES_DIR)}")
    
    # 1. Файлуудыг цуглуулах
    csv_files = [f for f in glob.glob(os.path.join(FEATURES_DIR, "*.csv")) if not os.path.basename(f).startswith("~$")]
    xlsx_files = [f for f in glob.glob(os.path.join(FEATURES_DIR, "*.xlsx")) if not os.path.basename(f).startswith("~$")]
    
    root_csv = glob.glob(os.path.join(os.path.dirname(FEATURES_DIR), "*.csv"))
    root_xlsx = glob.glob(os.path.join(os.path.dirname(FEATURES_DIR), "*.xlsx"))
    
    csv_files.extend([f for f in root_csv if os.path.abspath(f) != os.path.abspath(MERGED_FILE) and not os.path.basename(f).startswith("~$")])
    xlsx_files.extend([f for f in root_xlsx if not os.path.basename(f).startswith("~$")])
    
    all_files = list(set(csv_files + xlsx_files))
    print(f"Found {len(all_files)} total data files.")
    
    if not all_files:
        print("No data files found to combine!")
        return

    dfs = []
    for file_path in all_files:
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path, encoding="utf-8")
            else:
                df = pd.read_excel(file_path)
            
            # Баганыг цэгцлэх
            df = normalize_columns(df)
            
            if "label" in df.columns:
                dfs.append(df)
                print(f"Loaded & Normalized: {os.path.basename(file_path)} ({len(df)} rows)")
            else:
                print(f"Skipped (no 'label' column after normalization): {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

    if not dfs:
        print("No valid data frames found.")
        return

    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Баганын дарааллыг баталгаажуулах
    final_cols = [c for c in STD_COLS if c in merged_df.columns]
    merged_df = merged_df[final_cols]
    
    merged_df.to_csv(MERGED_FILE, index=False, encoding="utf-8")
    print("=" * 30)
    print(f"SUCCESS: Merged {len(merged_df)} rows into {MERGED_FILE}")
    print(f"Columns in merged file: {list(merged_df.columns)}")
    print("=" * 30)


if __name__ == "__main__":
    combine_data()
