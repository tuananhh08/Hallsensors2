import numpy as np
import pandas as pd
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent          
DATASET_DIR = BASE_DIR.parent / "Dataset"   

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")

# Calib params 
calib_df = pd.read_csv(DATASET_DIR / "Calibration_GRID_Offset_Sens.csv")
calib_df = calib_df.sort_values("sensor_index").reset_index(drop=True)
Ns = len(calib_df)

VQ_arr   = calib_df["offset_a_V"].to_numpy()        # (64,)
SENS_arr = calib_df["gain_g_V_per_T"].to_numpy()    # (64,)

print(f"  Sensors        : {Ns}")
print(f"  V_Q   range    : [{VQ_arr.min():.4f}, {VQ_arr.max():.4f}] V")
print(f"  SENS  range    : [{SENS_arr.min():.4f}, {SENS_arr.max():.4f}] V/T")

# Voltage thu được từ sensor
V_measured = pd.read_csv(
    DATASET_DIR / "grid_calib_data.csv", header=None
).values  # (5010, 64)

N = V_measured.shape[0]
assert V_measured.shape[1] == Ns, \
    f"V_measured có {V_measured.shape[1]} cột, calib có {Ns} sensors"

print(f"  Samples        : {N}")
print(f"  V range        : [{V_measured.min():.4f}, {V_measured.max():.4f}] V")

# ─── Tính ngược Bz = (V - V_Q) / SENS ────────────────────────────────────────
print("\nConverting Voltage -> Bz...")

Bz = (V_measured - VQ_arr[None, :]) / SENS_arr[None, :]  # (5010, 64)

print(f"\n─── Kết quả ─────────────────────────────────────────")
print(f"  Samples        : {N:,}")
print(f"  Bz range       : [{Bz.min():.6e}, {Bz.max():.6e}] T")
print(f"  Bz mean        : {Bz.mean():.6e} T")
print(f"─────────────────────────────────────────────────────")

# ─── Lưu kết quả ─────────────────────────────────────────────────────────────
out_path = DATASET_DIR / "Bgrid_calib.csv"  
pd.DataFrame(Bz).to_csv(out_path, header=False, index=False)
print(f"\nSaved -> {out_path}  ({N} rows x {Ns} cols)")