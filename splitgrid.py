"""
Cách dùng:
  python split_grid.py \
      --label Grid_points_coordinates.csv \
      --out_dir ./ckpt \
      --nx 4 --ny 4 --nz 2 \
      --target_test 150

Tham số:
  --nx / --ny / --nz   Số ô chia theo mỗi trục (default 4×4×2)
  --target_test        Số điểm test mong muốn (~100-200), script tự chọn
                       số block để đạt gần con số này nhất
  --corner             Chọn block từ góc (default) hoặc --no-corner để
                       chọn block biên ngẫu nhiên
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ─── Args ─────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--label",       default="Grid_points_coordinates.csv")
p.add_argument("--out_dir",     default="./ckpt")
p.add_argument("--nx",          type=int,   default=4)
p.add_argument("--ny",          type=int,   default=4)
p.add_argument("--nz",          type=int,   default=2)
p.add_argument("--target_test", type=int,   default=150,
               help="So diem test mong muon (~100-200)")
p.add_argument("--corner",      action="store_true", default=True,
               help="Chon block o goc khong gian (default)")
p.add_argument("--no-corner",   dest="corner", action="store_false",
               help="Chon block bien ngau nhien thay vi goc")
p.add_argument("--seed",        type=int,   default=42)
args = p.parse_args()


# ─── Đọc label ────────────────────────────────────────────────────────────────
def _read(path):
    df = pd.read_csv(path, header=None)
    try:
        df.iloc[0].astype(float)
        has_header = False
    except (ValueError, TypeError):
        has_header = True
    if has_header:
        df = pd.read_csv(path, header=0)
    return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

print(f"Reading {args.label} ...")
label_df = _read(args.label)
assert label_df.shape[1] == 5, f"Label can 5 cols, co {label_df.shape[1]}"
labels = label_df.values.astype(np.float32)   # (N, 5)
N      = len(labels)
xyz    = labels[:, :3]                         # (N, 3)
print(f"  Total points : {N}")
print(f"  X range : [{xyz[:,0].min():.4f}, {xyz[:,0].max():.4f}]")
print(f"  Y range : [{xyz[:,1].min():.4f}, {xyz[:,1].max():.4f}]")
print(f"  Z range : [{xyz[:,2].min():.4f}, {xyz[:,2].max():.4f}]")


# ─── Gán mỗi điểm vào block (ix, iy, iz) ─────────────────────────────────────
def assign_blocks(xyz, nx, ny, nz):
    """Trả về mảng (N, 3) chứa block index (ix, iy, iz) của từng điểm."""
    mins  = xyz.min(axis=0)
    maxs  = xyz.max(axis=0)
    spans = maxs - mins

    block_ids = np.zeros((len(xyz), 3), dtype=np.int32)
    for dim, n in enumerate([nx, ny, nz]):
        if spans[dim] < 1e-9:          # trục suy biến (tất cả cùng giá trị)
            block_ids[:, dim] = 0
        else:
            normalized         = (xyz[:, dim] - mins[dim]) / spans[dim]
            block_ids[:, dim]  = np.clip((normalized * n).astype(int), 0, n - 1)
    return block_ids

block_ids = assign_blocks(xyz, args.nx, args.ny, args.nz)

# Tạo dict: (ix,iy,iz) → list of sample indices
from collections import defaultdict
block_map = defaultdict(list)
for i, bid in enumerate(block_ids):
    block_map[tuple(bid)].append(i)

all_blocks  = list(block_map.keys())
block_sizes = {b: len(block_map[b]) for b in all_blocks}
print(f"\n  Grid     : {args.nx} × {args.ny} × {args.nz} = "
      f"{args.nx*args.ny*args.nz} cells")
print(f"  Non-empty: {len(all_blocks)} cells  "
      f"(avg {N/len(all_blocks):.1f} pts/cell)")


# ─── Chọn block làm test ──────────────────────────────────────────────────────
def corner_score(block_key, nx, ny, nz):
    """
    Điểm "góc-ness": block càng gần góc của lưới thì score càng cao.
    Dùng để ưu tiên chọn block ở góc xa nhất, tránh bao quanh bởi train.
    """
    ix, iy, iz = block_key
    # Khoảng cách đến góc gần nhất (chuẩn hoá về [0,1])
    dx = min(ix, nx - 1 - ix) / max(nx - 1, 1)
    dy = min(iy, ny - 1 - iy) / max(ny - 1, 1)
    dz = min(iz, nz - 1 - iz) / max(nz - 1, 1)
    # Score thấp = gần góc hơn → sắp xếp tăng dần
    return dx + dy + dz

def select_test_blocks(block_map, block_sizes, nx, ny, nz,
                       target_test, use_corner, seed):
    """
    Chọn tập block liền kề (connected region) làm test sao cho
    tổng số điểm ≈ target_test.
    Chiến lược: bắt đầu từ block góc, mở rộng BFS sang block kề
    cho đến khi đạt đủ số điểm.
    """
    rng = np.random.default_rng(seed)

    if use_corner:
        # Sắp xếp block theo corner_score (thấp = gần góc)
        sorted_blocks = sorted(all_blocks, key=lambda b: corner_score(b, nx, ny, nz))
        seed_block    = sorted_blocks[0]
    else:
        # Chọn ngẫu nhiên trong các block biên
        border = [b for b in all_blocks
                  if b[0] in (0, nx-1) or b[1] in (0, ny-1) or b[2] in (0, nz-1)]
        seed_block = tuple(rng.choice(border))

    # BFS mở rộng từ seed_block
    def neighbors(b):
        ix, iy, iz = b
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    nb = (ix+dx, iy+dy, iz+dz)
                    if nb in block_map:
                        yield nb

    selected = {seed_block}
    frontier = list(neighbors(seed_block))
    total    = block_sizes[seed_block]

    while frontier and total < target_test:
        # Chọn block frontier gần góc nhất (giữ vùng test compact)
        if use_corner:
            frontier.sort(key=lambda b: corner_score(b, nx, ny, nz))
            nxt = frontier.pop(0)
        else:
            nxt = frontier.pop(int(rng.integers(len(frontier))))

        if nxt in selected:
            continue
        selected.add(nxt)
        total += block_sizes[nxt]
        for nb in neighbors(nxt):
            if nb not in selected and nb not in frontier:
                frontier.append(nb)

    return selected, total

test_blocks, n_test_approx = select_test_blocks(
    block_map, block_sizes,
    args.nx, args.ny, args.nz,
    args.target_test, args.corner, args.seed)

# Lấy indices
test_idx  = []
for b in test_blocks:
    test_idx.extend(block_map[b])
test_idx  = np.array(sorted(test_idx))
train_idx = np.array([i for i in range(N) if i not in set(test_idx)])

print(f"\n  Selected test blocks : {len(test_blocks)}")
print(f"  Test  points : {len(test_idx):,}  "
      f"({100*len(test_idx)/N:.1f}%)")
print(f"  Train points : {len(train_idx):,}  "
      f"({100*len(train_idx)/N:.1f}%)")

# Kiểm tra: không có điểm test nào bị "bao vây" hoàn toàn bởi train
# (block liền kề của vùng test chỉ tiếp giáp biên hoặc block test khác)
inner_test_blocks = set()
for b in test_blocks:
    nb_list = []
    ix, iy, iz = b
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue
                nb = (ix+dx, iy+dy, iz+dz)
                if nb in block_map and nb not in test_blocks:
                    nb_list.append(nb)   # láng giềng là train
    if not nb_list:
        inner_test_blocks.add(b)         # hoàn toàn nằm trong vùng test

print(f"\n  Spatial leakage check:")
print(f"  Test blocks tiếp giáp train : "
      f"{len(test_blocks) - len(inner_test_blocks)} / {len(test_blocks)}")
print(f"  (Đây là các block biên của vùng test — chấp nhận được)")
print(f"  Test blocks hoàn toàn nằm trong vùng test (isolated): "
      f"{len(inner_test_blocks)}")


# ─── Lưu split_block.json ─────────────────────────────────────────────────────
os.makedirs(args.out_dir, exist_ok=True)
split_path = os.path.join(args.out_dir, "split_block.json")
with open(split_path, "w") as f:
    json.dump({
        "train":       train_idx.tolist(),
        "test":        test_idx.tolist(),
        "test_blocks": [list(b) for b in test_blocks],
        "grid":        {"nx": args.nx, "ny": args.ny, "nz": args.nz},
        "seed":        args.seed,
        "note":        "Spatial block holdout — test region is a contiguous "
                       "corner block, not randomly scattered.",
    }, f, indent=2)
print(f"\n  Split saved -> {split_path}")


# ─── Visualize ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5))

# ── Subplot 1: 3D scatter train vs test ───────────────────────────────────────
ax1 = fig.add_subplot(121, projection="3d")
train_xyz = xyz[train_idx]
test_xyz  = xyz[test_idx]

ax1.scatter(train_xyz[:, 0], train_xyz[:, 1], train_xyz[:, 2],
            c="#4472C4", s=8, alpha=0.4, label=f"Train ({len(train_idx)})")
ax1.scatter(test_xyz[:, 0],  test_xyz[:, 1],  test_xyz[:, 2],
            c="#FF0000", s=20, alpha=0.9, label=f"Test ({len(test_idx)})", marker="^")

ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.set_title("Spatial Block Holdout\n(red = test region)", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True)

# ── Subplot 2: 2D projection XY — hiển thị block boundaries ──────────────────
ax2 = fig.add_subplot(122)

mins_xyz = xyz.min(axis=0)
maxs_xyz = xyz.max(axis=0)
span_xyz = maxs_xyz - mins_xyz

# Vẽ tất cả điểm
ax2.scatter(train_xyz[:, 0], train_xyz[:, 1],
            c="#4472C4", s=8, alpha=0.3, label="Train")
ax2.scatter(test_xyz[:, 0],  test_xyz[:, 1],
            c="#FF0000", s=20, alpha=0.9, label="Test", marker="^")

# Vẽ đường lưới block trên mặt phẳng XY
if span_xyz[0] > 1e-9:
    for ix in range(args.nx + 1):
        xv = mins_xyz[0] + ix * span_xyz[0] / args.nx
        ax2.axvline(xv, color="gray", linewidth=0.5, alpha=0.5)
if span_xyz[1] > 1e-9:
    for iy in range(args.ny + 1):
        yv = mins_xyz[1] + iy * span_xyz[1] / args.ny
        ax2.axhline(yv, color="gray", linewidth=0.5, alpha=0.5)

# Tô màu nhẹ các block test trên mặt phẳng XY
import matplotlib.patches as mpatches
cell_w = span_xyz[0] / args.nx if span_xyz[0] > 1e-9 else 0.01
cell_h = span_xyz[1] / args.ny if span_xyz[1] > 1e-9 else 0.01
drawn  = set()
for b in test_blocks:
    xy_key = (b[0], b[1])
    if xy_key in drawn:
        continue
    drawn.add(xy_key)
    rx = mins_xyz[0] + b[0] * cell_w
    ry = mins_xyz[1] + b[1] * cell_h
    rect = mpatches.Rectangle((rx, ry), cell_w, cell_h,
                               linewidth=1, edgecolor="red",
                               facecolor="red", alpha=0.12)
    ax2.add_patch(rect)

ax2.set_xlabel("X"); ax2.set_ylabel("Y")
ax2.set_title(f"XY projection  |  Grid {args.nx}×{args.ny}\n"
              f"Red region = test blocks", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(False)
ax2.set_aspect("auto")

plt.suptitle(
    f"Grid Block Holdout Split  —  "
    f"Train: {len(train_idx)} pts  |  Test: {len(test_idx)} pts  "
    f"({100*len(test_idx)/N:.1f}%)",
    fontsize=12, y=1.01
)
plt.tight_layout()

vis_path = os.path.join(args.out_dir, "split_visualization.png")
plt.savefig(vis_path, dpi=150, bbox_inches="tight")
print(f"  Visualization saved -> {vis_path}")
plt.show()

print("\nDone. Run train.py và test.py như bình thường.")
print(f"train.py sẽ đọc split_block.json và chỉ dùng {len(train_idx)} điểm train.")
print(f"test.py  sẽ đọc split_block.json và đánh giá trên {len(test_idx)} điểm test.")