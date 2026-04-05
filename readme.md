# VRU Trajectory Prediction at Smart Intersections

Diploma thesis — Trajectory prediction of Vulnerable Road Users (VRUs) at smart intersections for road safety and autonomous driving applications.

## Dataset

Uses the [IMPTC Dataset](https://github.com/kav-institute/imptc-dataset) (IEEE IV 2023):
- 270 sequences recorded at a smart inner-city intersection in Germany
- 2,500+ VRU trajectories (pedestrians, cyclists, e-scooter riders, strollers)
- 20,000+ vehicle trajectories
- LiDAR + camera sensors @ 25 Hz
- Additional context: weather, traffic light signals, ground segmentation

Download from [Zenodo](https://zenodo.org/records/14811016).
Ground plane segmentation map: [Google Drive](https://drive.google.com/file/d/1uPcvJv-etmImUCoVugBAJUJmFrLcd5kR/view)

---

## Pipeline

### Step 1 — Download & Load
```bash
python load_imptc_trajectories.py   # sample dataset (347MB)
python load_full_dataset.py         # full trajectory dataset (2.9GB)
python download_and_build.py        # full sequences (17GB)
```

### Step 2 — Temporal Matching
```bash
python matching.py
```
Finds all VRU pairs that co-exist in time across train/eval/test splits.
Output: `matches_train.json`, `matches_eval.json`, `matches_test.json`

### Step 3 — Relative Geometry
```bash
python relative_motion.py
```
For each pedestrian A at every timestamp @ 10Hz:
- Downsamples 25Hz → 10Hz
- Computes heading θ_A from consecutive positions
- Rotation matrix R_θA:
  - `p_B|A = R_θA · (p_B - p_A)` — relative position
  - `V_rel = R_θA · (v_B - v_A)` — relative velocity

### Step 4 — 5×5 Local Grid
```bash
python build_grid_dataset.py
```
For each pedestrian A at every timestamp @ 10Hz:
- A is always at origin **(0, 0)**
- 5×5 grid, cell size **1m × 1m** → covers 5m × 5m area
- 7 channels: VRU occupation, vehicle occupation, ground type, rel_x, rel_y, rel_vx, rel_vy
- Output: `results/grid_dataset.json` (1,920 pedestrians, 570,902 frames)

### Step 5 — Ground Plane Segmentation
```bash
python build_grid_with_ground.py
```
Adds ground type to each grid cell using KDTree lookup on the segmentation map.
Ground is **North-Up** (no rotation with pedestrian heading).

| ID | Name |
|----|------|
| 0 | road |
| 1 | sidewalk |
| 2 | ground |
| 3 | curb |
| 4 | road_line |
| 5 | crosswalk |
| 6 | bikelane |
| 7 | unknown |

### Step 6 — Traffic Lights
```bash
python add_traffic_lights.py
```
Adds pedestrian signal status (f1, f2, f3) to each timestamp @ 10Hz.
Uses nearest-neighbor interpolation from 1Hz → 10Hz.
Uses `results/interactions_detailed.csv` for scene→archive mapping.

| Code | State |
|------|-------|
| 4 | green |
| 10 | red |
| 20 | yellow |
| 30 | red-yellow |
| 2 | yellow-blinking |
| 11 | disabled |

### Step 7 — Training Samples
```bash
python build_training_samples.py
```
Sliding window over each trajectory:
- **Input**: 38 frames (3.8 sec) → 5×5 grid + traffic lights
- **Output**: 48 frames (4.8 sec) → (x, y) positions

| File | Shape | Size |
|------|-------|------|
| `train_X.npy` | (332146, 38, 7, 5, 5) | 8.8 GB |
| `train_X_tl.npy` | (332146, 38, 3) | 151 MB |
| `train_Y.npy` | (332146, 48, 2) | 127 MB |
| `eval_X.npy` | (37794, 38, 7, 5, 5) | 1.0 GB |
| `eval_X_tl.npy` | (37794, 38, 3) | 17 MB |
| `eval_Y.npy` | (37794, 48, 2) | 14 MB |
| `test_X.npy` | (38871, 38, 7, 5, 5) | 1.0 GB |
| `test_X_tl.npy` | (38871, 38, 3) | 17 MB |
| `test_Y.npy` | (38871, 48, 2) | 14 MB |

### Step 8 — Pedestrian Context Export
```bash
python export_pedestrian_context.py
```
Exports for every pedestrian at every timestamp:
- Global position (x, y)
- Ground type under their feet (center cell [2,2])
- Traffic light states (f1, f2, f3)

Output: `results/pedestrian_context.csv`

### Step 9 — Baseline Model
```bash
python baseline_cv_new.py
```
Constant Velocity baseline: predicts future positions using last observed velocity.

---

## Mathematical Framework

For each pedestrian A at timestamp t:

**Rotation matrix** (global → local frame of A):
```
R_θA = [[cos θ_A,  sin θ_A],
        [-sin θ_A, cos θ_A]]
```

**Relative position** of B in A's local frame:
```
p_B|A = R_θA · (p_B - p_A)
```

**Relative velocity**:
```
V_rel = R_θA · (v_B - v_A)
```

---

## Grid Representation

```
5×5 grid around pedestrian A @ each timestamp:

        W ←————————————→ E
    N   ┌────┬────┬────┬────┬────┐
    ↑   │    │ V  │    │    │    │  V = vehicle
        ├────┼────┼────┼────┼────┤
        │    │    │ P  │    │    │  P = pedestrian
        ├────┼────┼────┼────┼────┤
        │ SW │ SW │ 🚶 │ RD │ RD │  SW=sidewalk, RD=road
        ├────┼────┼────┼────┼────┤
        │ SW │    │    │    │    │
    ↓   ├────┼────┼────┼────┼────┤
    S   │ CW │ CW │    │    │    │  CW = crosswalk
        └────┴────┴────┴────┴────┘

🚶 = A always at (0,0) — center cell [2,2]
Each cell = 1m × 1m — total area: 5m × 5m
Grid is North-Up (global orientation)
```

---

## Baseline Results

### Constant Velocity Model

**Eval set** (37,794 samples):

| Metric | Value |
|--------|-------|
| ADE | 0.5341 m |
| FDE | 1.3869 m |

**By ground type:**

| Ground Type | ADE | FDE | n |
|-------------|-----|-----|---|
| sidewalk | 0.5379 | 1.3986 | 28,648 |
| crosswalk | 0.5154 | 1.3381 | 6,349 |
| road | 0.4080 | 1.0651 | 1,221 |
| curb | 0.6593 | 1.6514 | 842 |
| bikelane | 1.2065 | 2.8334 | 42 |

**By traffic light f1:**

| State | ADE | FDE | n |
|-------|-----|-----|---|
| green | 0.4068 | 1.0122 | 16,134 |
| red | 0.6290 | 1.6659 | 21,660 |

---

## Result Files

### results/

| File | Description |
|------|-------------|
| `pedestrian_context.csv` | Every pedestrian at every timestamp with position, ground type and traffic lights |
| `baseline_cv_eval_results.csv` | Constant velocity baseline results on eval set |
| `baseline_cv_test_results.csv` | Constant velocity baseline results on test set |
| `interactions_detailed.csv` | All VRU-VRU and VRU-vehicle interaction pairs |
| `interactions_summary.csv` | Summary statistics of interactions per scene |
| `pedestrian_math_dataset_v2_full.csv` | Full pedestrian interaction dataset with relative geometry for all 270 scenes |
| `pedestrian_math_with_map_and_lights.csv` | Interaction dataset with ground plane + traffic light state |
| `dataset_relative_geometry.csv` | Relative geometry matched from trajectory dataset |
| `relative_geometry_detailed.csv` | Detailed relative geometry with rotation matrices |
| `src_mapping.csv` | Mapping from sample_id to source scene and archive |
| `crosswalk_approaching_vehicle_cases.csv` | Cases where vehicle is approaching crosswalk with VRU present |

---

## Scripts

| Script | Description |
|--------|-------------|
| `load_imptc_trajectories.py` | Download & load sample dataset (347MB) |
| `load_full_dataset.py` | Download & load full trajectory dataset (2.9GB) |
| `download_and_build.py` | Download all 17GB sequence archives |
| `matching.py` | Temporal matching of co-existing VRU pairs |
| `relative_motion.py` | Compute relative position & velocity per timestamp |
| `build_grid_dataset.py` | Build 5×5 local grid for each pedestrian timestamp |
| `build_grid_with_ground.py` | Add ground plane segmentation (North-Up) |
| `add_traffic_lights.py` | Add pedestrian traffic light signals (f1, f2, f3) |
| `build_training_samples.py` | Create sliding window training samples (38→48 frames) |
| `build_dataset.py` | Build relative geometry dataset from trajectory JSON |
| `export_pedestrian_context.py` | Export position + ground type + traffic lights per timestamp |
| `enrich_crosswalk_cases.py` | Enrich crosswalk cases with full trajectory and context |
| `baseline_cv_new.py` | Constant velocity baseline with ADE/FDE metrics |

---

## Data Summary

| Split | Tracks | Frames @10Hz | Samples |
|-------|--------|--------------|---------|
| Train | 1,536  | ~457K | 332,146 |
| Eval  | 192    | ~57K  | 37,794  |
| Test  | 192    | ~57K  | 38,871  |

**Where pedestrians are (ground type):**

| Ground Type | Frames | % |
|-------------|--------|---|
| sidewalk | 456,970 | 80.0% |
| crosswalk | 81,515 | 14.3% |
| road | 13,153 | 2.3% |
| curb | 10,830 | 1.9% |

**Traffic light f1:**

| State | Frames | % |
|-------|--------|---|
| red | 308,930 | 54.1% |
| green | 261,972 | 45.9% |

---

## Requirements

```bash
pip install numpy scipy requests tqdm
```

---

## References

- [IMPTC Dataset Paper (IEEE IV 2023)](https://doi.org/10.1109/IV55152.2023.10186776)
- [Dataset Download (Zenodo)](https://zenodo.org/records/14811016)
- [Ground Plane Segmentation Map](https://drive.google.com/file/d/1uPcvJv-etmImUCoVugBAJUJmFrLcd5kR/view)
