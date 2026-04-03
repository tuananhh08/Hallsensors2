# import matplotlib.pyplot as plt
# import numpy as np
# # Data
# theory = [
# 1.5821017524458345,1.5893426624269396,1.5910239897187832,1.5907029260992982,1.5982166137296077,1.5900367601515966,1.585319912095583,1.5852420851817295,1.579442610785615,1.5741366794131701,1.5748583703629935,1.580134982820555,1.5955401761564534,1.5994228597017357,1.5760359698595034,1.5761381087422017,1.586878094361791,1.5784345311081367,1.602885885987369,1.5918805191487986,1.591127792793066,1.5853129620628992,1.6032037300055593,1.5912969048202406,1.5877103524458678,1.5822995516946234,1.5754599433957894,1.5843857772455416,1.5892728321519498,1.5864654120499972,1.5785088168071955,1.5793890091086151,1.5781627912181084,1.5790504084311043,1.587135875925836,1.5992560375061298,1.5875505050695562,1.589156034632168,1.5769278583031707,1.5979480918060176,1.5806047281805402,1.5879512651762822,1.576266567616857,1.5993466094594202,1.605767446844848,1.5860967820048533,1.5744151607797827,1.583462128974961,1.5844307028855158,1.5857982624538205,1.5809189338999818,1.5884455692171588,1.5849517836724656,1.5762290447203209,1.5812152803510624,1.5762003535108047,1.5881686711748129,1.5768992414005647,1.5746349837114502,1.5829366483859768,1.601473169135193,1.5886516754508073,1.6019025676217789,1.579340078713237

# ]
# print(np.max(theory))
# measured = [
# 1.5844,1.592,1.5942,1.5932,1.5998,1.5916,1.587,1.5868,1.582,1.5762,1.5774,1.582,1.5974,1.6016,1.5778,1.5778,1.589,1.5808,1.6056,1.594,1.593,1.5874,1.605,1.5932,1.5898,1.5848,1.578,1.587,1.5916,1.5886,1.5804,1.5814,1.5804,1.5818,1.5892,1.6016,1.5894,1.5912,1.579,1.6,1.5834,1.5904,1.5784,1.602,1.6078,1.5888,1.5768,1.5854,1.5866,1.5876,1.583,1.591,1.5876,1.5786,1.5836,1.5786,1.5906,1.5788,1.577,1.5858,1.604,1.591,1.6042,1.582

# ]
# print()
# # X axis (index từ 1 -> 64)
# x = list(range(1, 65))

# # Plot
# plt.figure()
# plt.plot(x, theory, marker='o', label='Theory')
# plt.plot(x, measured, marker='x', label='Measured')
# plt.xlabel('Sensor index')
# plt.ylabel('Voltage')

# error = [m - t for m, t in zip(measured, theory)]

# plt.legend()
# plt.figure()
# plt.plot(x, error, marker='o')
# plt.title('Error (Measured - Theory)')
# plt.xlabel('Index')
# plt.ylabel('Error')
# plt.show()

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "InvProb"

print("Loading data...")
 
V_compute     = pd.read_csv(DATASET_DIR / "Vgrid_calib.csv",     header=None).values  # (5010, 64)
V_measurement = pd.read_csv(DATASET_DIR / "grid_calib_data.csv", header=None).values  # (5010, 64)
coords        = pd.read_csv(DATASET_DIR / "Grid_points_coordinates.csv")               # có header
 
assert V_compute.shape == V_measurement.shape, \
    f"Shape không khớp: Vcompute {V_compute.shape} vs Vmeasure {V_measurement.shape}"
 
N, Ns = V_compute.shape
print(f"  Shape          : {N} điểm x {Ns} sensors")
 
# ─── Tính error ───────────────────────────────────────────────────────────────
error = np.abs(V_compute - V_measurement)       # (5010, 64)
 
mae_global     = error.mean()
mae_per_point  = error.mean(axis=1)             # (5010,)
mae_per_sensor = error.mean(axis=0)             # (64,)
mse            = np.mean(error ** 2)
error_min      = error.min()
error_max      = error.max()
 
# ─── Threshold ────────────────────────────────────────────────────────────────
THRESHOLD = 0.1
 
# Điểm có ít nhất 1 sensor có |Vcompute - Vmeasure| > threshold
point_mask = (error > THRESHOLD).any(axis=1)   # (5010,) bool
n_over     = point_mask.sum()
 
# Trong số đó, bao nhiêu điểm có z < -0.136
z_values = coords.loc[point_mask, "z"].values
n_low_z  = np.sum(z_values < -0.136)
 
# ─── Kết quả ──────────────────────────────────────────────────────────────────
print(f"\n─── Kết quả so sánh Vcompute vs Vmeasurement ────────────")
print(f"  MAE toàn bộ                  : {mae_global:.6f} V")
print(f"  MAE per point  (mean)        : {mae_per_point.mean():.6f} V  "
      f"[min={mae_per_point.min():.6f}, max={mae_per_point.max():.6f}]")
print(f"  MAE per sensor (mean)        : {mae_per_sensor.mean():.6f} V  "
      f"[min={mae_per_sensor.min():.6f}, max={mae_per_sensor.max():.6f}]")
print(f"  MSE                          : {mse:.6f} V²")
print(f"  Error min                    : {error_min:.6f} V")
print(f"  Error max                    : {error_max:.6f} V")
print(f"  Số điểm có error > {THRESHOLD} V  : {n_over:,} / {N:,} "
      f"({100 * n_over / N:.2f}%)")
print(f"  Số điểm error > {THRESHOLD} V "
      f"và z < -0.136 : {n_low_z:,} / {n_over:,}")
print(f"─────────────────────────────────────────────────────────")