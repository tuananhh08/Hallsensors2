import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "InvProb"

# ─── Load data ────────────────────────────────────────────────────────────────
V_compute     = pd.read_csv(DATASET_DIR / "Vgrid_calib.csv",     header=None).values
V_measurement = pd.read_csv(DATASET_DIR / "grid_calib_data.csv", header=None).values
coords        = pd.read_csv(DATASET_DIR / "Grid_points_coordinates.csv")
sensor_pos    = pd.read_csv(DATASET_DIR / "sensors_position_calib.csv").values  # (64, 3)

error         = np.abs(V_compute - V_measurement)   # (5010, 64)
mae_per_point = error.mean(axis=1)                  # (5010,)
max_per_point = error.max(axis=1)                   # (5010,)

THRESHOLD = 0.08
point_mask = (error > THRESHOLD).any(axis=1)

xyz    = coords[["x", "y", "z"]].values
z      = xyz[:, 2]
mask_z = z < -0.136

# ─── 1. Phân phối error theo z ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Phân tích error theo vị trí không gian", fontsize=13)

ax = axes[0, 0]
sc = ax.scatter(z, mae_per_point, c=mae_per_point, cmap="RdYlGn_r",
                s=6, alpha=0.6, vmin=0, vmax=0.1)
ax.axvline(-0.136, color="red", lw=1.2, linestyle="--", label="z = -0.136")
ax.axhline(THRESHOLD, color="orange", lw=1.2, linestyle="--", label=f"threshold = {THRESHOLD}")
ax.set_xlabel("z (m)"); ax.set_ylabel("MAE (V)")
ax.set_title("MAE vs z"); ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax)

# ─── 2. Scatter 3D projection: XY colored by error ───────────────────────────
ax = axes[0, 1]
sc2 = ax.scatter(xyz[:, 0], xyz[:, 1], c=mae_per_point,
                 cmap="RdYlGn_r", s=6, alpha=0.6, vmin=0, vmax=0.1)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("MAE trên mặt phẳng XY"); plt.colorbar(sc2, ax=ax)

# ─── 3. Scatter XZ colored by error ─────────────────────────────────────────
ax = axes[0, 2]
sc3 = ax.scatter(xyz[:, 0], xyz[:, 2], c=mae_per_point,
                 cmap="RdYlGn_r", s=6, alpha=0.6, vmin=0, vmax=0.1)
ax.axhline(-0.136, color="red", lw=1.2, linestyle="--", label="z = -0.136")
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
ax.set_title("MAE trên mặt phẳng XZ"); ax.legend(fontsize=8)
plt.colorbar(sc3, ax=ax)

# ─── 4. Khoảng cách min đến sensor vs error ──────────────────────────────────
r_min_per_point = np.linalg.norm(
    xyz[:, None, :] - sensor_pos[None, :, :], axis=2).min(axis=1)  # (5010,)

ax = axes[1, 0]
ax.scatter(r_min_per_point, mae_per_point, c=mask_z.astype(int),
           cmap="bwr", s=6, alpha=0.5)
ax.axhline(THRESHOLD, color="orange", lw=1.2, linestyle="--")
ax.set_xlabel("Khoảng cách min đến sensor (m)")
ax.set_ylabel("MAE (V)")
ax.set_title("MAE vs khoảng cách min đến sensor\n(đỏ = z < -0.136)")

# ─── 5. Histogram MAE: z < -0.136 vs z >= -0.136 ────────────────────────────
ax = axes[1, 1]
ax.hist(mae_per_point[mask_z],  bins=40, alpha=0.6, label="z < -0.136",  color="red")
ax.hist(mae_per_point[~mask_z], bins=40, alpha=0.6, label="z >= -0.136", color="blue")
ax.axvline(THRESHOLD, color="orange", lw=1.2, linestyle="--", label="threshold")
ax.set_xlabel("MAE (V)"); ax.set_ylabel("Số điểm")
ax.set_title("Phân phối MAE theo vùng z"); ax.legend(fontsize=8)

# ─── 6. Error theo sensor index — trung bình riêng 2 nhóm ───────────────────
ax = axes[1, 2]
mae_sensor_lowz  = error[mask_z,  :].mean(axis=0)   # (64,)
mae_sensor_highz = error[~mask_z, :].mean(axis=0)   # (64,)
sensor_idx = np.arange(64)
ax.bar(sensor_idx, mae_sensor_lowz,  alpha=0.6, label="z < -0.136",  color="red",  width=0.8)
ax.bar(sensor_idx, mae_sensor_highz, alpha=0.6, label="z >= -0.136", color="blue", width=0.8)
ax.axhline(THRESHOLD, color="orange", lw=1.2, linestyle="--", label="threshold")
ax.set_xlabel("Sensor index"); ax.set_ylabel("MAE trung bình (V)")
ax.set_title("MAE theo sensor: so sánh 2 nhóm z"); ax.legend(fontsize=8)

plt.tight_layout()
out_path = BASE_DIR / "error_analysis.jpg"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_path}")

# ─── In thống kê tóm tắt ─────────────────────────────────────────────────────
print(f"\n─── Thống kê theo vùng z ─────────────────────────────────")
for label, mask in [("z < -0.136", mask_z), ("z >= -0.136", ~mask_z)]:
    n   = mask.sum()
    mae = mae_per_point[mask].mean()
    pct = point_mask[mask].sum()
    r   = r_min_per_point[mask].mean()
    print(f"  {label:15s} | n={n:4d} | MAE={mae:.5f} V | "
          f"error>threshold={pct:3d} ({100*pct/n:.1f}%) | "
          f"r_min avg={r*100:.2f} cm")

print(f"\n─── Tương quan ───────────────────────────────────────────")
corr_z = np.corrcoef(z, mae_per_point)[0, 1]
corr_r = np.corrcoef(r_min_per_point, mae_per_point)[0, 1]
print(f"  Pearson corr(z, MAE)          : {corr_z:+.4f}")
print(f"  Pearson corr(r_min, MAE)      : {corr_r:+.4f}")
print(f"─────────────────────────────────────────────────────────")