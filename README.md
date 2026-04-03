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

---

## Pipeline

### 1. Download & Load
```bash
python load_imptc_trajectories.py   # sample dataset (347MB)
python load_full_dataset.py         # full trajectory dataset (2.9GB)
python download_and_build.py        # full sequences (17GB)
```

### 2. Temporal Matching
```bash
python matching.py
```
Finds all VRU pairs that co-exist in time across train/eval/test splits.

### 3. Relative Geometry
```bash
python relative_motion.py
```
For each pedestrian A at every timestamp:
- Downsamples 25Hz → 10Hz
- Computes heading θ_A from consecutive positions
- Applies rotation matrix R_θA:
  - `p_B|A = R_θA · (p_B - p_A)` — relative position
  - `V_rel = R_θA · (v_B - v_A)` — relative velocity

### 4. 5×5 Local Grid
```bash
python build_grid_dataset.py
```
For each pedestrian A at every timestamp @ 10Hz:
- A is always at origin (0, 0)
- 5×5 grid, cell size 1m × 1m → covers 5m × 5m area
- 7 channels per cell: VRU occupation, vehicle occupation, ground type, rel_x, rel_y, rel_vx, rel_vy

### 5. Ground Plane Segmentation
```bash
python build_grid_with_ground.py
```
Adds ground type to each grid cell using KDTree lookup on the segmentation map (North-Up, no rotation).

Ground types: `road(0), sidewalk(1), ground(2), curb(3), road_line(4), crosswalk(5), bikelane(6), unknown(7)`

### 6. Traffic Lights
```bash
python add_traffic_lights.py
```
Adds pedestrian signal status (f1, f2, f3) to each timestamp:
- `green=4, red=10, yellow=20, red-yellow=30, yellow-blinking=2`
- Nearest-neighbor interpolation from 1Hz → 10Hz

### 7. Training Samples
```bash
python build_training_samples.py
```
Sliding window over each trajectory:
- **Input**: 38 frames (3.8 sec) → grid + traffic lights
- **Output**: 48 frames (4.8 sec) → (x, y) positions

Output numpy arrays:
```
train_X.npy     (N, 38, 7, 5, 5)  — input grid
train_X_tl.npy  (N, 38, 3)        — input traffic lights
train_Y.npy     (N, 48, 2)        — output positions
```
NOT GET TRAINING (απλα εχω τα νουμερα)
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

🚶 = A always at center (0,0)
Each cell = 1m × 1m
```

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
| `build_grid_with_ground.py` | Add ground plane segmentation to grid |
| `add_traffic_lights.py` | Add pedestrian traffic light signals (f1,f2,f3) |
| `build_training_samples.py` | Create sliding window training samples (38→48 frames) |
| `build_dataset.py` | Build relative geometry dataset from trajectory JSON |

---

## Result Files

### results/

| File | Description |
|------|-------------|
| `interactions_detailed.csv` | All VRU-VRU and VRU-vehicle interaction pairs with timestamps, distance, overlap frames |
| `interactions_summary.csv` | Summary statistics of interactions per scene |
| `pedestrian_math_dataset_v2_full.csv` | Full pedestrian interaction dataset with relative geometry (p_B\|A, V_rel, heading, interaction zone) for all 270 scenes |
| `pedestrian_math_dataset_v2.csv` | Subset of above (sample only) |
| `pedestrian_math_dataset.csv` | Earlier version of math dataset |
| `pedestrian_math_with_map.csv` | Math dataset enriched with ground plane type per interaction |
| `pedestrian_math_with_map_and_lights.csv` | Math dataset enriched with ground plane + traffic light state |
| `dataset_relative_geometry.csv` | Relative geometry (rel_x, rel_y, rel_vx, rel_vy) matched from trajectory dataset |
| `relative_geometry_detailed.csv` | Detailed relative geometry with rotation matrices (Rga, Rba) |
| `pedestrian_moments_summary.csv` | Summary of key pedestrian moments (crossing, stopping, etc.) |
| `pedestrian_moments_neighbors.csv` | Neighbor VRUs at each key pedestrian moment |
| `src_mapping.csv` | Mapping from sample_id to source scene (scene_path, archive) |
| `matched_codes.csv` | Matched track codes between trajectory dataset and sequences |
| `sample_codes.csv` | Sample ID codes for train/eval/test |
| `archive_index.csv` | Index of all scenes across all archive files |
| `local_scene_0004_with_map.csv` | Local scene export for scene 0004 with map overlay |
| `close_crosswalk_vehicle_cases.csv` | Cases where VRUs are close to crosswalk with nearby vehicle |
| `crosswalk_approaching_vehicle_cases.csv` | Cases where vehicle is approaching crosswalk with VRU present |
| `extreme_crosswalk_approaching_cases.csv` | Extreme/high-risk crosswalk approach cases |
| `extreme_front_approaching_crosswalk_cases.csv` | Extreme front-approach crosswalk cases |
| `top_extreme_cases.csv` | Top ranked extreme interaction cases |

### preprocessed/

| File | Description |
|------|-------------|
| `ALL_VRUs.csv` | All VRU tracks with positions and velocities |
| `ALL_VRUs_FINAL.csv` | Final cleaned version of all VRU tracks |
| `train.csv` / `eval.csv` | Raw train/eval split trajectories |
| `train_FINAL.csv` / `eval_FINAL.csv` | Final cleaned train/eval trajectories |
| `train_COMPLETE.csv` / `eval_COMPLETE.csv` | Complete trajectories with all features |
| `train_with_gt.csv` / `eval_with_gt.csv` | Trajectories with ground truth future positions |
| `trajectories_FULL.csv` | Full trajectory dataset (all splits combined) |
| `interactions.csv` | Interaction pairs (basic) |
| `interactions_detailed.csv` | Interaction pairs with full geometry |
| `interactions_summary.csv` | Interaction statistics summary |
| `social_interactions.csv` | Social force model interactions between VRUs |
| `matches_all.csv` | All temporal matches across splits |
| `statistics.csv` | Dataset statistics (counts, durations, speeds) |
| `baseline_cv_results.csv` | Baseline constant velocity model prediction results |

---

## Data Summary

| Split | Tracks | Frames @10Hz |
|-------|--------|--------------|
| Train | 1,536  | ~457K        |
| Eval  | 192    | ~57K         |
| Test  | 192    | ~57K         |

| VRU Class  | Train | Eval | Test |
|------------|-------|------|------|
| Pedestrian | 2,628 | 310  | 843  |
| Cyclist    | 617   | 58   | 179  |
| Motorcycle | 233   | 22   | 103  |
| Scooter    | 85    | 9    | 14   |
| Stroller   | 22    | 1    | 9    |

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
